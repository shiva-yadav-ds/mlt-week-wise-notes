# MLT Notes — Week 1–12

Simple static notes site with week navigation, topic indexes and System / Light / Dark themes. System is the default; manual preferences are remembered in the browser.

## Deploy

Upload the contents of `mlt-notes-deploy.zip` to the root of a new GitHub repository (include the `.github` folder). The archive contains the deployable project, original week notes and images, without local Git metadata or source PDFs. Keep your local PDFs as reference.

### Vercel

1. Import that GitHub repository in Vercel.
2. Keep Root Directory as the repository root (`.`), not `notes-site`.
3. Deploy. `vercel.json` installs the Python Markdown dependency, generates all 12 pages, checks links, and serves `notes-site/dist`.

### GitHub Pages

1. Use `main` as the repository default branch (or change the workflow branch).
2. Open Settings → Pages → Build and deployment → Source → GitHub Actions.
3. Push to `main`, or run “Deploy notes to GitHub Pages” from Actions.

All page and image URLs are relative, so project paths such as `username.github.io/repository/` work. Deploy only `notes-site/dist`; it contains rendered notes and images. GitHub Pages is public: choose a repository visibility/hosting plan suitable for your notes.

## Edit and preview

Edit the original `week1` through `week12` Markdown files and images. There is no second content copy to maintain.

```sh
python3 -m pip install -r requirements.txt
python3 notes-site/build.py
python3 notes-site/check.py
node notes-site/check-theme.cjs
python3 -m http.server 8000 --directory notes-site/dist
```

Open http://localhost:8000. Node is only used for the theme test; Python builds the site. CSS, theme script and MathJax are in `notes-site/dist`, which must be kept when rebuilding. Commit note changes and generated pages together; both deployment workflows rebuild from the Markdown sources.

Original reference PDFs are retained locally. Only byte-identical website content copies were removed. Local `notes-site/.git` belongs to the earlier Sites deployment; use the clean ZIP for a new GitHub repository, not that nested Git repository.

Configuration references: [Vercel](https://vercel.com/docs/project-configuration/vercel-json), [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
# mlt-week-wise-notes
# mlt-week-wise-notes
