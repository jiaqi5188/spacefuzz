"""Last line of defence for the pages: an unexpected error shows a short message with the
details folded underneath, instead of a raw stack trace across the page.

st.stop() and st.rerun() raise BaseException subclasses (ScriptControlException), which
`except Exception` lets through, so page control flow is untouched."""
import sys
import traceback

import streamlit as st

MESSAGE = ("Something unexpected went wrong on this page. Your check-in and the cached "
           "results are unchanged: reload the page, or open the Dashboard from the menu.")


def run(page) -> None:
    try:
        page()
    except Exception as exc:            # noqa: BLE001 - shown to the user, logged in full
        traceback.print_exc(file=sys.stderr)
        st.warning(MESSAGE, icon=":material/error:")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}", language=None)
