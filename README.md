## EasyModel

EasyModel is a **visual no-code fine‑tuning studio** for Hugging Face models.

It lets you build end‑to‑end training workflows on a canvas, start jobs with a single click, and inspect rich analytics for every run – all from your browser. The backend runs remotely (e.g. Colab / cloud), so you never have to install anything locally.

---

## What you can do

- **Fine‑tune Hugging Face models without code**
  - Pick a model and dataset by URL (e.g. `openai-community/gpt2`, `rajpurkar/squad`).
  - Choose a task type (generation, classification, seq2seq, token classification).
  - Set epochs, batch size, max length, subset size, text/label fields.
  - Hit “Train Model” and watch it go.

- **Design training pipelines visually**
  - Drag the following nodes onto a canvas:
    - **HuggingFace Model** – which model to fine‑tune.
    - **HuggingFace Dataset** – which dataset to use.
    - **Finetuning** – hyperparameters and run configuration.
  - Connect nodes to define the flow (model → finetune, dataset → finetune).
  - Zoom, pan, rearrange, and organize your experiments visually.

- **Track progress in real time**
  - Live status messages during training (via server‑sent events).
  - Current epoch and total epochs.
  - Progress indicators directly inside the Finetuning node.
  - Clear success / error states when a job completes or fails.

- **Explore analytics after each run**
  - **Perplexity**:
    - Training vs validation vs baseline curves across epochs.
    - Quick sense of over/under‑fitting and improvements over the base model.
  - **GQS (Generative Quality Score)**:
    - Radar chart over fluency, coherence, grammar, relevance.
    - Overall GQS score + improvement vs baseline.
  - **Semantic Consistency & Drift**:
    - Similarity over segments to see how stable your model behaves.
  - **Token Efficiency & Compression Ratio**:
    - Compare pre‑trained vs fine‑tuned token efficiency across tasks.
    - Scatter plot of original vs generated length with a diagonal reference.

- **Manage multiple projects**
  - Create and switch between projects from the left sidebar.
  - Each project has its **own React Flow canvas** and analytics history.
  - The currently active project is clearly highlighted in the UI.

---

## Experience highlights

### Visual React Flow canvas

The core experience is a **React Flow‑powered graph**:

- Nodes represent models, datasets, and finetuning schemas.
- Edges represent data flow between them.
- You can:
  - Drag nodes from the palette.
  - Connect them to define which model uses which dataset.
  - Open the Finetuning node to configure training.
- The canvas is **per‑project and persistent**:
  - Your graph is automatically saved per `projectId`.
  - You can navigate to analytics, other pages, or refresh the browser – when you come back, your nodes, edges, and their internal data are still there.

### Dark mode, everywhere

The entire app supports **light and dark mode**:

- A theme toggle lives in the sidebar.
- Charts, nodes, text, backgrounds, and UI components all adapt to the active theme.
- The design intentionally avoids harsh contrast or unreadable states in either mode.

### Analytics views that actually match the training

Analytics aren’t fake placeholders – they are computed during your actual training runs:

- The backend records and aggregates metrics at the epoch level.
- The frontend analytics page reads those metrics and:
  - Shows **per‑epoch** curves for training and validation perplexity.
  - Computes **improvement percentages** over baseline.
  - Updates GQS and semantic metrics to match your latest run.
- If for any reason an SSE update doesn’t carry analytics, the frontend has a **fallback fetch** once training completes.

---

## How it works (high level)

- **Frontend**
  - Next.js app (deployed on Vercel) provides:
    - React Flow canvas with per‑project persistence (using Zustand).
    - Analytics dashboard with Recharts‑based visualizations.
    - tRPC‑driven data layer and Prisma‑backed database for projects and analytics.
  - State is carefully managed so:
    - Node/edge changes are saved atomically to localStorage and synced to the DB.
    - Only the active project’s graph is touched at any time.
    - No timers or intervals are owned by pages – they live in the store instead.

- **Backend**
  - A FastAPI service (e.g. running in Colab or another host) handles:
    - Fine‑tuning jobs using Hugging Face Transformers and PyTorch.
    - Progress streaming via Server‑Sent Events.
    - On‑the‑fly analytics:
      - Perplexity for language models.
      - Semantic similarity and coherence.
      - GQS metrics.
      - Token efficiency and compression ratios.
  - When a job runs, the backend:
    - Streams `stage`, `epoch`, `progress`, and messages.
    - Emits analytics payloads that are saved and later visualized.

---

## Privacy & security

- **Hugging Face token**:
  - Entered in the Finetuning node when you start a job.
  - Stored only in `window.sessionStorage` for the browser session.
  - Not written to the database or committed to Git.
- **Environment variables**:
  - Backend and frontend secrets live in `.env` files **outside** version control.
  - The repository’s `.gitignore` aggressively ignores `.env`, `*.db`, and other sensitive file types.
- **Open‑source ready**:
  - No hard‑coded API keys or access tokens in the repo.
  - No backdoor network calls beyond the documented backend and Hugging Face endpoints.

---

## Status and future directions

EasyModel is now in a state that is:

- **Stable enough for people to try**.
- Architected for **commercial‑style usage** (clear separation of frontend/backend, no secrets in code, durable canvas state).
- Friendly to future contributions:
  - The codebase is organized by feature (React Flow, analytics, APIs).
  - The React Flow persistence layer centers on a single Zustand store rather than scattered effects.

If you’re interested in extending it (e.g. new analytics, additional task types, model registries, or multi‑backend support), the existing structure is built to support that growth.

License
This project is licensed under the MIT License.