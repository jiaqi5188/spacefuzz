# AI disclosure

SpaceFuzz (NASA Space Apps Challenge 2026, Team Raccoon) was built with the help of the AI tools listed below. This file states where each one was used, as the Space Apps rules ask for AI-generated code, text and audio to be disclosed.

## Tools and where they were used

| Tool | Used for | Where it shows |
| --- | --- | --- |
| Claude (Anthropic) | Planning, research support and writing: execution plan, data-analysis plan, summaries of NASA risk pages and standards, drafts of the README and `TECHNICAL.md` | Documentation |
| Claude Code (Anthropic) | Writing the code and tests (data readers, plan adjustment, Streamlit interface), code review, data checks, screenshots, the video recording and assembly scripts (`video/`), the video thumbnail (composed from an app screenshot) | The whole code base |
| ElevenLabs | The voice-over of the 30-second demo video | The video carries a visible "AI voice-over (ElevenLabs)" watermark for its whole length; its title / comment metadata and the YouTube description state that the voice-over is AI-generated |
| OpenAI API (optional) | An off-by-default hint on the check-in page: when the data-quality gate rejects a check-in, a user who supplies their own `OPENAI_API_KEY` can ask for a short plain-language explanation | `ui/assist.py`; it sends only the rejected fields, the values entered and their plausible ranges |

## What the AI tools did not do

- **No AI in the safety decisions.** The risk numbers, ratings, rule checks and plan adjustments come from deterministic code with a fixed random seed. The optional OpenAI hint never changes a decision or a number, and without a key the app runs exactly the same. It was not used in the demo or the recordings.
- **No hand-typed results.** The demo numbers come from the engine output (`cache/`), not from text written by a person or a model.
- **No AI-generated images.** The video uses a NASA photograph (NASA/JPL-Caltech/MSSS/TAMU, PIA17936) and an Unsplash photograph; all other images are screenshots of the app.

## Human review

The team reviewed the changes and checked the facts and numbers against the sources and the engine output. Responsibility for the content stays with the team.
