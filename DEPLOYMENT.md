# Deployment Guide

This guide explains how to deploy EasyModel to Vercel for public access.

## Important Notes

**Backend Limitation**: The Python FastAPI backend with ML dependencies (PyTorch, Transformers) cannot run on Vercel due to:
- Serverless function size limits (50MB compressed)
- ML models are too large for serverless functions
- Cold start times would be excessive

**Solution**: Deploy the frontend to Vercel and the backend separately to a service that supports Python with large dependencies.

## Deployment Steps

### Step 1: Deploy Frontend to Vercel

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Import project in Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import your GitHub repository
   - **Important**: Set the **Root Directory** to `grizzly`

3. **Configure Build Settings**
   - Framework Preset: Next.js (auto-detected)
   - Root Directory: `grizzly`
   - Build Command: `npm run build` (default)
   - Output Directory: `.next` (default)
   - Install Command: `npm install` (default)

4. **Set Environment Variables**
   In Vercel project settings, add these environment variables:
   
   ```
   NEXT_PUBLIC_EASYMODEL_API_URL=https://your-backend-url.com
   DATABASE_URL=your-database-url
   ```
   
   **For Database Options:**
   - **Option A (Recommended for Production)**: Use Vercel Postgres
     - Go to Vercel dashboard → Storage → Create Database
     - Select Postgres
     - Copy the connection string to `DATABASE_URL`
   
   - **Option B (Development)**: Keep SQLite (not recommended for production)
     - Use `file:./prisma/dev.db` (but this won't persist across deployments)

5. **Deploy**
   - Click "Deploy"
   - Wait for build to complete
   - Your frontend will be live at `https://your-project.vercel.app`

### Step 2: Deploy Backend

Choose one of these options:

#### Option A: Railway (Recommended)

1. Go to [railway.app](https://railway.app)
2. Create new project → Deploy from GitHub
3. Select your repository
4. Railway will auto-detect Python
5. Set environment variables:
   ```
   HUGGINGFACE_API_KEY=your-token
   ```
6. Railway will automatically build and deploy
7. Copy the public URL (e.g., `https://your-app.railway.app`)
8. Update `NEXT_PUBLIC_EASYMODEL_API_URL` in Vercel to point to this URL

#### Option B: Render

1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt` (create requirements.txt)
   - **Start Command**: `python main.py` or `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3
5. Set environment variables:
   ```
   HUGGINGFACE_API_KEY=your-token
   PORT=10000
   ```
6. Deploy and copy the public URL
7. Update frontend environment variable

#### Option C: Docker + Any Platform

1. Create `Dockerfile` in project root:
   ```dockerfile
   FROM python:3.12-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8000
   CMD ["python", "main.py"]
   ```

2. Create `requirements.txt`:
   ```txt
   fastapi
   uvicorn[standard]
   python-dotenv
   transformers
   torch
   datasets
   huggingface-hub
   ```

3. Deploy to any Docker-compatible platform (Fly.io, DigitalOcean, etc.)

### Step 3: Update CORS Settings

After deploying the backend, update `main.py` to allow your Vercel domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://your-project.vercel.app",  # Add your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Step 4: Update Frontend Environment Variable

1. Go to Vercel dashboard → Your Project → Settings → Environment Variables
2. Update `NEXT_PUBLIC_EASYMODEL_API_URL` to your deployed backend URL
3. Redeploy the frontend (or it will auto-deploy on next push)

## Post-Deployment Checklist

- [ ] Frontend is accessible at Vercel URL
- [ ] Backend is accessible and returns health check
- [ ] CORS is configured correctly
- [ ] Environment variables are set in both frontend and backend
- [ ] Database is accessible (if using external database)
- [ ] Hugging Face API key is set in backend
- [ ] Test creating a project in the frontend
- [ ] Test connecting nodes
- [ ] Test starting a training job

## Troubleshooting

### Frontend shows "Failed to fetch"
- Check that `NEXT_PUBLIC_EASYMODEL_API_URL` is set correctly
- Verify backend is running and accessible
- Check CORS settings in backend

### Backend deployment fails
- Ensure Python 3.12 is specified
- Check that all dependencies are in requirements.txt
- Verify environment variables are set

### Database errors
- If using SQLite, it won't persist on Vercel (use Postgres)
- Check DATABASE_URL is correct
- Run migrations: `npx prisma migrate deploy`

## Production Considerations

1. **Use Vercel Postgres** instead of SQLite for production
2. **Set up proper error monitoring** (Sentry, etc.)
3. **Configure rate limiting** on backend
4. **Use environment-specific API keys**
5. **Set up CI/CD** for automatic deployments
6. **Monitor backend resource usage** (ML training is resource-intensive)

## Getting Your Public Link

Once deployed:
- **Frontend**: Your Vercel project URL (e.g., `https://easymodel.vercel.app`)
- **Backend**: Your Railway/Render/etc. URL (e.g., `https://easymodel-api.railway.app`)

Share the frontend URL with users - it's your public link!

