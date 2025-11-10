# MedGemma: API & CLI

This repository contains the backend API, modern web frontend, and command-line interface (CLI) for MedGemma, a tool designed to interact with medical language models.

## Project Structure

This project contains three main components. Please refer to the `README.md` within each directory for specific setup and usage instructions.

-   [`/medgemma-api`](./medgemma-api/): The backend server built with Python (FastAPI).
-   [`/medgemma-frontend`](./medgemma-frontend/): Modern React-based web interface with shadcn/ui components.
-   [`/medgemma-cli`](./medgemma-cli/): The command-line tool to do local inferencing.

## Getting Started

### Clone the Repository with Submodules

This project uses git submodules for the frontend. Clone with:

```bash
git clone --recurse-submodules git@github.com:MarsPresLai/MedGemma-Toolkit.git
```

If you already cloned without submodules, initialize them:

```bash
git submodule update --init --recursive
```

### Setup Frontend

1. Navigate to the frontend directory:
   ```bash
   cd medgemma-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Build the frontend for production:
   ```bash
   npm run build
   ```

   Or run in development mode (with hot reload):
   ```bash
   npm run dev
   ```

### Setup Backend

1. Navigate to the API directory:
   ```bash
   cd medgemma-api
   ```

2. Follow the instructions in [`medgemma-api/README.md`](./medgemma-api/README.md) to set up and run the backend server.

The backend will automatically serve the built frontend from `medgemma-frontend/dist/` when you access the root URL.