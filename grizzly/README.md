# Eregion

## Overview

This project is a web application that utilizes various technologies such as React, Next.js, Prisma, and PostgreSQL, along with a set of UI components for building a user-friendly interface. The application features a dashboard where users can manage projects, view analytics, and take advantage of machine learning models via integration with Hugging Face.

## Major Parts of the Code

1. **Database Schema (Prisma)**: 
   - The database schema defines models such as `User`, `Project`, `Analytics`, and `Job`, which represent the core entities of the application. Each model has its own fields, relationships, and constraints.

2. **Authentication**:
   - The application uses Clerk for user authentication, enabling sign-in and sign-out functionalities.

3. **Client-Side Components**:
   - Components like `ThemeProvider`, `LeftSidebar`, `RightSidebar`, and `DockComponent` manage the layout and theme of the application.
   - The `NodePalette` component allows users to drag and drop different node types (e.g., HuggingFace models, datasets, finetuning nodes) onto a canvas.

4. **Charts and Analytics**:
   - The application provides various visualizations, including radar charts, line charts, and bar charts that represent model performance metrics, such as accuracy and perplexity.

5. **React Flow**:
   - The application uses `@xyflow/react` for building a flow diagram, allowing users to connect different nodes representing models and datasets visually.

6. **Dialogs and Popovers**:
   - UI components like `Dialog` and `Popover` are utilized for displaying additional information and confirmations, enhancing user interaction.

## Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/project-name.git
   cd project-name
   ```

2. **Install Dependencies**:
   - Ensure you have Node.js installed. Then, run:
   ```bash
   npm install
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory and add the following variables:
   ```env
   DATABASE_URL=your_database_url
   CLERK_API_KEY=your_clerk_api_key
   ```

4. **Initialize the Database**:
   - Run the Prisma migration command to set up the database:
   ```bash
   npx prisma migrate dev --name init
   ```

5. **Run the Application**:
   ```bash
   npm run dev
   ```

6. **Visit the Application**:
   - Open your browser and go to `http://localhost:3000`.

## Usage Instructions

- **Authentication**:
  - Use the Sign In button to log in. If you don’t have an account, follow the prompts to create one.

- **Project Management**:
  - Use the left sidebar to create, edit, or delete projects. You can also view project details in the main content area.

- **Analytics Visualization**:
  - Navigate through different analytics tabs to view performance metrics related to your projects.

- **Node Management**:
  - Drag and drop nodes from the Node Palette into the canvas area to create a workflow involving various models and datasets.

- **Settings**:
  - You can adjust your settings in the user profile section as needed.

## Conclusion

This application provides a comprehensive platform for managing machine learning models and projects. With its user-friendly design and robust functionality, users can easily interact with various features to visualize and analyze their data. 

Feel free to explore and contribute to the project!
