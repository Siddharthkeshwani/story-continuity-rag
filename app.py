"""Story Continuity Assistant for Episodic Writers

Project title — Story Continuity Assistant for Episodic Writers
Problem statement — Writers preparing a later episode or recap need to check what
happened and what each character knew at the time. Later revelations can spoil an earlier scene.
Proposed solution — Search dialogue up to the selected episode, check each claim,
cite the exchange and identify what the available passages leave unresolved.
Target users — Writers, script editors and adaptation teams working on episodic audio fiction.
Data or documents — English Campaign 1 and Campaign 2 transcripts from CRD3 and reviewed editorial
facts linked to original turns. Aligned summaries help locate evaluation evidence.
Expected output — A short answer or verdict per claim, with campaign, episode,
original turn range and an expandable English source excerpt.
"""
import os
import streamlit as st
from src.config import CHAT_KEY, CHAT_MODEL, CHAT_PROVIDER, corpus, excerpt, manifest
from src.artifacts import ensure_assets
from src.retrieve import backend_store, check_story
from src.v1 import QUESTION

st.set_page_config(page_title="Story continuity", page_icon=":material/menu_book:")
st.title("Story continuity")
st.write("Check a story detail against the dialogue available to your writer.")
try:
    ensure_assets()
    meta = manifest()
    backend_store("v2", meta["fingerprint"])
except Exception as err:
    st.error("The transcript or index bundle could not be loaded. Check the dataset configuration and release checksum.")
    st.stop()
revision = meta["fingerprint"]
campaigns = meta["campaigns"]
if not campaigns:
    st.warning("This edition has no eligible episodes.")
    st.stop()
with st.sidebar:
    st.header("Source material")
    campaign = st.selectbox("Campaign", sorted(map(int,campaigns)), format_func=lambda n:f"Campaign {n}")
    episodes = campaigns[str(campaign)]
    cutoff = st.selectbox("Use episodes through", episodes, key=f"cutoff_{campaign}", format_func=lambda n:f"Episode {n}")
    st.caption(f"{len(episodes)} episodes in this campaign · {len(meta['episode_ids'])} in the corpus")
    st.divider()
    language = st.selectbox("Answer language", ["English", "Hindi", "Hinglish"])
    version = st.selectbox("Search method", ["v4", "v3", "v2"],
                           format_func=lambda v:{"v4":"V4 · Reranked search","v3":"V3 · Chroma","v2":"V2 · FAISS"}[v])
    has_chat = bool(os.getenv(CHAT_KEY))
    evidence_only = st.toggle("Show search results only", value=not has_chat)
    st.caption(f"Answer model · {CHAT_PROVIDER} · {CHAT_MODEL}")
    st.caption("Original dialogue is shown in English. Speaker labels follow the source transcript.")
    st.markdown("[CRD3 source and licence](https://github.com/RevanthRameshkumar/CRD3)")
question = st.text_area("Story question or recap claim", value=QUESTION if campaign == 1 else
                        "Which town borders the Ustaloch in the opening narration?", key=f"question_{campaign}", height=95)
context = (revision, campaign, cutoff, version, language, question, evidence_only, CHAT_MODEL)
if st.session_state.get("context") != context:
    st.session_state.pop("result", None)
    st.session_state.context = context
if st.button("Check story", type="primary", icon=":material/search:"):
    st.session_state.pop("result", None)
    if not question.strip():
        st.warning("Enter a story question or a recap claim.")
    else:
        try:
            with st.spinner("Searching the eligible episodes…"):
                st.session_state.result = check_story(question.strip(), version, campaign, cutoff, revision, language, evidence_only)
        except FileNotFoundError as err:
            st.error(f"The search index or source files are unavailable. {err}")
        except Exception as err:
            if "timeout" in type(err).__name__.lower():
                st.error("The chat service timed out. Retry the same question.")
            elif "connection" in type(err).__name__.lower():
                st.error(f"The {CHAT_PROVIDER} API could not be reached. Retry or use search results only.")
            else:
                if getattr(err, "status_code", None) == 429:
                    st.error("The model provider rejected the request because of a rate or credit limit.")
                elif getattr(err, "status_code", None) == 404:
                    st.error(f"{CHAT_PROVIDER} did not make {CHAT_MODEL} available for this request. Use search results only to inspect the original passages.")
                elif isinstance(err, (ValueError, RuntimeError)) and type(err).__module__ == "builtins":
                    st.error(str(err))
                else:
                    st.error(f"The check could not finish ({type(err).__name__}).")
result = st.session_state.get("result")
if result:
    tables = corpus(revision)
    def display_source(cid):
        passage = tables["chunks"][cid]
        with st.expander(f"{passage['episode_id']} · turns {passage['turn_start']}–{passage['turn_end']} · {cid}"):
            if passage["document_type"] == "editorial":
                st.caption("An editor's fact led to this exchange.")
            for turn in excerpt(passage, tables["turns"]):
                st.text(f"{turn['turn_no']}  {', '.join(turn['speakers'])}  {turn['text']}")
    if not result["final"]:
        st.info("No passages are available for this campaign and cutoff.")
    elif evidence_only:
        st.subheader("Ranked passages")
        for candidate in result["final"]:
            display_source(candidate["id"])
    else:
        st.subheader(f"Continuity check through episode {cutoff}")
        for part in result["answer"]["parts"]:
            st.caption(part["status"].capitalize())
            st.write(part["claim"])
            for quote in part.get("support_quotes", []):
                st.text(quote)
            for cid in dict.fromkeys(part["evidence_ids"]):
                display_source(cid)
    with st.expander("Search details"):
        st.caption(f"{version.upper()} · Retrieval {result['retrieval_ms']:.0f} ms · Reranking {result['rerank_ms']:.0f} ms")
        st.write("Initial order", [p["id"] for p in result["initial"]])
        st.write("Final order", [p["id"] for p in result["final"]])

