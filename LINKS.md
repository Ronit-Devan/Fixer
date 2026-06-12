# ET Production URLs

- **Web app:** https://et-web-eight.vercel.app
- **API:** https://devan-p-et-api.hf.space
- **HF Space:** https://huggingface.co/spaces/devan-p/et-api
- **Repo:** https://github.com/devan-p/ET

## Local dev

cd packages/web/api && uv run uvicorn et_api.server:app --reload --port 8000
cd packages/web/app && pnpm dev

## Deploy

# Backend
HF_TOKEN=hf_xxx ./packages/web/api/deploy-hf.sh

# Frontend
cd packages/web/app && vercel --prod
