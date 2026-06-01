# Neural Notes — Deep Learning Blog

A clean, fast blog built with Jekyll for GitHub Pages, focused on deep learning fundamentals.

## 🚀 Deploy to GitHub Pages (5 minutes)

### 1. Create a GitHub repo

Go to github.com → New repository → name it `yourusername.github.io`  
(Replace `yourusername` with your actual GitHub username)

### 2. Upload this folder

```bash
git init
git add .
git commit -m "Initial blog setup"
git remote add origin https://github.com/yourusername/yourusername.github.io.git
git push -u origin main
```

### 3. Enable GitHub Pages

- Go to your repo → **Settings** → **Pages**
- Source: **Deploy from a branch**
- Branch: `main` / `/(root)`
- Click **Save**

Your site will be live at `https://yourusername.github.io` within 1–2 minutes!

---

## ✏️ Customize It

### Change your info: `_config.yml`

```yaml
title: "Your Blog Name"
tagline: "Your tagline here"
author:
  name: "Your Name"
  github: "yourusername"
  twitter: "yourhandle"
  bio: "Your one-line bio here."
url: "https://yourusername.github.io"
```

### Write a new post

Create a file in `_posts/` named `YYYY-MM-DD-post-title.md`:

```markdown
---
layout: post
title: "Your Post Title"
date: 2025-07-01
category: "Fundamentals"
tags: [neural-networks, math]
read_time: 8
excerpt: "One sentence summary of your post."
---

Your content here. Supports Markdown, code blocks, and math!

Inline math: $E = mc^2$

Display math:
$$\nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}$$

Code:
```python
import torch
x = torch.tensor([1.0])
```
```

---

## 🧪 Run Locally (Optional)

```bash
# Install Ruby + Bundler, then:
bundle install
bundle exec jekyll serve
# Visit http://localhost:4000
```

---

## 📁 Structure

```
├── _config.yml          # Site configuration
├── _layouts/
│   ├── default.html     # Base layout (header, footer)
│   └── post.html        # Blog post layout
├── _includes/
│   └── sidebar.html     # Sidebar (author, tags, recent posts)
├── _posts/              # Your blog posts go here
│   └── YYYY-MM-DD-title.md
├── assets/
│   └── css/main.css     # All styles
├── index.html           # Homepage
├── blog.html            # Blog listing
├── tags.html            # Tags index
├── about.html           # About page
└── Gemfile              # Ruby dependencies
```

---

## ✨ Features

- **Math rendering** via KaTeX (use `$...$` and `$$...$$`)
- **Syntax highlighting** via Rouge
- **RSS feed** auto-generated
- **Tags & categories** with index page
- **Responsive** — works on mobile
- **Fast** — no JS frameworks, plain CSS
- **SEO-friendly** via jekyll-seo-tag

---

Built with [Jekyll](https://jekyllrb.com) · Hosted on [GitHub Pages](https://pages.github.com)
