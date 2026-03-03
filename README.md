# visual-novel-template

A minimal, client-side visual novel template you can clone and run anywhere. Write branching stories, add backgrounds and character portraits, and ship a playable prototype fast.

---

## Features

- Title and game flow
  - Title screen with Continue (auto-enabled if a save exists) and Start New Game
  - Persistent saving/loading via localStorage (scene, line index, flags, seen scenes)
- Dialogue system
  - Typewriter text with skip-to-full on click
  - Advance on click or Space/Enter
  - Speaker name injected into the dialogue box
- Scenes and assets
  - Scene model defined in code (Scenes object) with bg, portrait, music, and lines
  - Wipe-left scene transitions
  - Portrait appears after the wipe (nice polish)
- Choices and state
  - Choices overlay with options
  - Flags via set: { … } and branching using next: "sceneId"
  - Lines can compute text from state (functions allowed in text)
- Audio
  - Optional looping BGM per scene
  - Mute toggle (state persisted)
- Dev console (super helpful!)
  - Jump to scenes, skip to end, inspect current engine state
- Structure
  - Fully client-side, easy to deploy as a static site

---

## Project structure

```
.
├── assets/
│   ├── images/
│   │   ├── bg/        # Backgrounds
│   │   └── char/      # Character portraits
│   └── music/         # Background music
├── scripts/
│   └── script.js      # VN engine + demo content (Scenes)
├── styles/
│   └── style.css
├── index.html         # Open this to play
└── README.md
```

---

## Getting started

### Frontend only

- Quick start (no tooling)
  - Download or clone this repo
  - Open index.html in a modern browser

### Full app (recommended: live feeds + reader content)

1) Install deps

- npm install

2) Run the server

- npm run dev

Then visit http://localhost:3000

Notes:
- The backend serves the static UI + provides:
  - GET /api/items
  - GET /api/item/:id
- Data sources (MVP): CISA KEV JSON, NVD CVE 2.0 API (last 24h), The Hacker News RSS, arXiv cs.CR Atom.

---

## Add your story

- Edit scripts/script.js and modify the Scenes object:
  - Define scene ids, bg, portrait, music, and lines
  - Add choices with:
    - type: "choice"
    - prompt: "..."
    - options: [{ text, set: { flag: true }, next: "sceneId" }]
  - Lines can be functions to compute text from state:
    - text: (state) => state.flags.tookPark ? "..." : "..."

- Place assets
  - Backgrounds: assets/images/bg/
  - Portraits: assets/images/char/
  - Music: assets/music/

- Saving and settings
  - Game saves are stored under localStorage (key vn_save_v1)
  - Mute state is persisted (key vn_settings_v1)

- Dev console
  - Toggle via the button in the UI to jump scenes, skip to end, and inspect state

---

## Roadmap and project planning

For feature ideas and a planning starter, see INTRO.md. It contains a “Project kickoff” checklist and a roadmap of high-impact improvements.
