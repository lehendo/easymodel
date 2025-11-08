# EasyModel

EasyModel is a comprehensive no-code platform for fine-tuning and evaluating machine learning models, specifically designed for Hugging Face models and datasets. The platform combines a powerful FastAPI backend with an intuitive Next.js frontend featuring a visual workflow interface built with React Flow.

## Overview

EasyModel provides a complete solution for machine learning practitioners who want to fine-tune models without writing code. Users can visually connect model and dataset nodes, configure training parameters through an intuitive interface, and monitor training progress in real-time. The platform supports multiple task types including text generation, classification, sequence-to-sequence, and token classification.

## Architecture

The project consists of two main components:

**Backend (FastAPI)**: A Python-based REST API that handles model fine-tuning, progress tracking via Server-Sent Events (SSE), and comprehensive analytics. The backend integrates with Hugging Face Hub for model management and supports real-time training cancellation.

**Frontend (Next.js)**: A React-based web application featuring a visual node-based interface where users can drag and drop model and dataset nodes, connect them to a fine-tuning schema node, configure parameters, and initiate training with a single click. The frontend provides real-time progress updates, analytics visualization, and project management.

## Key Features

### Visual Workflow Interface
- Drag-and-drop node-based interface for creating training workflows
- Support for Hugging Face model and dataset nodes
- Automatic column inference from datasets
- Single-click training initiation without deployment steps
- Real-time connection status indicators

### Model Fine-tuning
- Support for multiple task types: generation, classification, sequence-to-sequence, token classification
- Configurable training parameters: epochs, batch size, sequence length, subset size
- Automatic dataset field detection
- Integration with Hugging Face Hub for model pushing
- Real-time progress tracking with progress bars and epoch information
- Training cancellation capability

### Analytics and Evaluation
- Comprehensive model analytics after training completion
- Perplexity analysis for language models
- Accuracy, F1 score, and loss metrics
- Automatic analytics card display upon training completion

### Project Management
- Multiple project support with persistent storage
- Local storage and database synchronization
- Project creation, editing, and deletion
- Automatic "Untitled Project" creation for new users

## Technology Stack

**Backend:** FastAPI, PyTorch, Transformers, Hugging Face Hub, Server-Sent Events, SQLite (via Prisma)

**Frontend:** Next.js 15, React 18, React Flow, tRPC, Prisma, Tailwind CSS, TypeScript

## Installation

### Prerequisites
- Python 3.12 (for backend ML dependencies)
- Node.js 18+ and npm
- Hugging Face API token

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/lehendo/easymodel.git
cd easymodel

# Create Python 3.12 virtual environment
python3.12 -m venv venv312
source venv312/bin/activate  # On Windows: venv312\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "HUGGINGFACE_API_KEY=your-huggingface-api-key-here" > .env
# Get your API token from https://huggingface.co/settings/tokens

# Run the backend server
python main.py
```

The backend will start on `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd grizzly

# Install dependencies
npm install

# Set up environment variables
echo "NEXT_PUBLIC_EASYMODEL_API_URL=http://localhost:8000" > .env.local
echo "DATABASE_URL=file:./prisma/dev.db" >> .env.local

# Generate Prisma client and run migrations
npx prisma generate
npx prisma migrate dev

# Start the development server
npm run dev
```

The frontend will start on `http://localhost:3000`

## Environment Variables

### Backend (.env)
- `HUGGINGFACE_API_KEY` or `HF_TOKEN`: Your Hugging Face API token (required for fine-tuning)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins (optional, defaults to localhost)

### Frontend (.env.local)
- `NEXT_PUBLIC_EASYMODEL_API_URL`: Backend API URL (default: http://localhost:8000)
- `DATABASE_URL`: SQLite database path (default: file:./prisma/dev.db)

**Security Note**: Never commit `.env` files to version control. They are automatically ignored by git.

## Usage

### Starting Training

1. Open the frontend application in your browser
2. Create a new project or select an existing one
3. Drag a "HuggingFace Model" node onto the canvas and enter a model name (e.g., "gpt2")
4. Drag a "HuggingFace Dataset" node onto the canvas and enter a dataset name (e.g., "squad")
5. Drag a "Finetuning Schema" node onto the canvas
6. Connect the model and dataset nodes to the finetuning node (top or bottom connectors)
7. Configure training parameters in the finetuning node
8. Click "Train Model" to start training

### Training Parameters

- **Output Model Name**: Name for your fine-tuned model on Hugging Face Hub
- **Epochs**: Number of training epochs
- **Batch Size**: Training batch size
- **Max Length**: Maximum sequence length
- **Subset Size**: Number of samples to use from the dataset
- **Task Type**: Type of ML task (generation, classification, seq2seq, token_classification)
- **Text Field**: Name of the text column in your dataset
- **Label Field**: Name of the label column (required for non-generation tasks)

### Monitoring Progress

During training, you'll see real-time progress bars, current epoch information, progress messages, and a cancel button. Upon completion, an analytics card will display relevant metrics.

## API Endpoints

### Fine-tuning
- `POST /finetuning/` - Start a fine-tuning job
- `GET /finetuning/progress/{job_id}` - Stream training progress (SSE)
- `POST /finetuning/cancel/{job_id}` - Cancel a running training job

### Analytics
- `POST /analytics/analytics` - Run comprehensive model analytics
- `POST /analytics/evaluate` - Evaluate a single model
- `POST /analytics/compare` - Compare two models

### Utility
- `GET /health` - Health check endpoint

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Quick Summary

**Frontend (Vercel)**: Push code to GitHub, import in Vercel with root directory set to `grizzly`, configure environment variables (`NEXT_PUBLIC_EASYMODEL_API_URL` and `DATABASE_URL`), and deploy.

**Backend (Railway/Render/etc.)**: Deploy Python backend to Railway, Render, or similar service, set `HUGGINGFACE_API_KEY`, update frontend's `NEXT_PUBLIC_EASYMODEL_API_URL` to point to backend URL, and configure backend CORS to allow your Vercel domain.

**Important**: The Python backend with ML dependencies cannot run on Vercel due to size limitations. Deploy it separately to a service that supports large Python applications.

## Project Structure

```
easymodel/
├── main.py                    # FastAPI backend entry point
├── requirements.txt           # Python dependencies
├── easymodel/                 # Backend package
│   ├── endpoints/            # API endpoints
│   └── utils/                 # Core logic and analytics
├── grizzly/                   # Next.js frontend
│   ├── src/                   # Source code
│   └── prisma/                # Database schema
└── README.md                  # This file
```

## Development

### Running Tests

```bash
# Backend tests
cd tests && python test_analytics.py

# Frontend type checking
cd grizzly && npm run typecheck
```

### Database Management

```bash
cd grizzly
npx prisma studio  # Open Prisma Studio
npx prisma migrate dev  # Create new migration
```

## Contributing

Contributions are welcome! Fork the repository, create a feature branch, make your changes, add tests if applicable, and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

Built with FastAPI, Next.js, React Flow, PyTorch, Transformers, and Hugging Face libraries.
