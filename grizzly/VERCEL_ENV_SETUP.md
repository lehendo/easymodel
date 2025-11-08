# Setting Up Environment Variables in Vercel

## Method 1: Vercel Dashboard (Recommended)

1. Go to your Vercel project dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add each variable manually:

### Required Variables:

**NEXT_PUBLIC_EASYMODEL_API_URL**
- Value: Your deployed backend URL (e.g., `https://your-app.railway.app`)
- Environment: Production, Preview, Development (select all)
- **Note**: Set this AFTER you deploy your backend

**DATABASE_URL**
- Value: Your database connection string
  - For Vercel Postgres: Get from Vercel Dashboard → Storage → Create Database → Postgres
  - Format: `postgresql://user:password@host:5432/database`
- Environment: Production, Preview, Development (select all)
- **Important**: For production, use Vercel Postgres, not SQLite

### Optional Variables (if needed):

- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` (if using Clerk)
- `CLERK_SECRET_KEY` (if using Clerk)

## Method 2: Vercel CLI

If you have the Vercel CLI installed:

```bash
# Install Vercel CLI (if not installed)
npm i -g vercel

# Login to Vercel
vercel login

# Add environment variables
vercel env add NEXT_PUBLIC_EASYMODEL_API_URL
vercel env add DATABASE_URL

# Pull environment variables to local .env.local (for local development)
vercel env pull .env.local
```

## Method 3: Copy from .env.vercel

1. Open `grizzly/.env.vercel` in this repository
2. Copy the variable names and values
3. Paste them into Vercel dashboard (Settings → Environment Variables)
4. **Replace placeholder values** with your actual values

## Important Notes

1. **NEXT_PUBLIC_EASYMODEL_API_URL**: 
   - Must be set AFTER deploying your backend
   - Should be the full URL (e.g., `https://your-backend.railway.app`)
   - Must include `https://` or `http://`

2. **DATABASE_URL**:
   - For production: Use Vercel Postgres (recommended)
   - SQLite (`file:./prisma/dev.db`) won't work in production on Vercel
   - Get Postgres URL from Vercel Dashboard → Storage

3. **After adding variables**:
   - Redeploy your application for changes to take effect
   - Or wait for the next automatic deployment

## Quick Checklist

- [ ] Deploy backend to Railway/Render/etc.
- [ ] Get backend URL
- [ ] Create Vercel Postgres database (or use existing)
- [ ] Add `NEXT_PUBLIC_EASYMODEL_API_URL` in Vercel dashboard
- [ ] Add `DATABASE_URL` in Vercel dashboard
- [ ] Redeploy frontend
- [ ] Test the application

