# bge-embed-rs UI design

Source of truth: the Penpot file (4 boards: `Status / Downloading`, `Status / Ready`,
`Connections`, `Settings`). Every value below is a design token; the egui theme must use
exactly these values and nothing else.

## Tokens

### Colour (semantic, per theme)

| Token | Light | Dark | egui use |
|---|---|---|---|
| `color.bg.window` | `#F7F8FA` | `#171A1F` | `panel_fill`, `window_fill`, input field bg |
| `color.bg.surface` | `#FFFFFF` | `#262B33` | cards, pills, secondary buttons, nav bar, `extreme_bg_color` |
| `color.border.default` | `#DCE0E6` | `#3F4652` | 1 px inner stroke on cards/pills/inputs/secondary buttons, progress track |
| `color.text.default` | `#171A1F` | `#F7F8FA` | primary text, `override_text_color` |
| `color.text.muted` | `#5B6472` | `#9AA3AF` | captions, hints, inactive tabs, toggle-off track, neutral status dot |
| `color.accent.default` | `#1F6FD1` | `#5AA2F5` | primary buttons, active tab, progress fill, toggle-on track, `selection.bg_fill`, `hyperlink_color` |
| `color.accent.on` | `#FFFFFF` | `#0F1115` | text on accent |
| `color.status.success` | `#15803D` | `#3CC47C` | Ready / Connected dot |
| `color.status.warning` | `#92600E` | `#E0A43A` | Downloading / needs-attention dot |
| `color.status.error` | `#B42318` | `#F06A62` | Failed dot + error text |

Theme follows the OS (`ctx.system_theme()` / eframe `follow_system_theme`).

### Spacing (4 px grid)

| Token | px | Used for |
|---|---|---|
| `spacing.gap.xs` | 4 | label↔value in stats, title↔subtitle, nav bar padding, tab gap |
| `spacing.gap.sm` | 8 | inside cards between rows of a card body, pill dot↔label |
| `spacing.gap.md` | 12 | between rows in a list card, between stat cards, row item gaps |
| `spacing.gap.lg` | 16 | between top-level sections on a screen |
| `spacing.inset.sm` | 8 | pill horizontal padding, input padding, tab padding, button vertical padding |
| `spacing.inset.md` | 16 | card padding, button horizontal padding |
| `spacing.inset.lg` | 24 | screen padding |

### Radius

| Token | px | Used for |
|---|---|---|
| `radius.control` | 4 | buttons, inputs, tabs, URL field |
| `radius.card` | 8 | cards, nav bar |
| (pill) | full | status pill, avatars (circle), toggles |

### Type — Inter (embed the font)

| Token | px | Weight usage |
|---|---|---|
| `font.size.caption` | 12 | captions, hints, button labels, tab labels, pill label |
| `font.size.body` | 14 | body, card titles (600), setting labels (500), tool names (600) |
| `font.size.subtitle` | 17 | stat values (600) |
| `font.size.title` | 20 | app title (600) |
| `font.size.display` | 24 | (reserved) |

Weights used: 400, 500, 600.

## Window

480 × 560 px, not resizable below that. Column layout, screen padding 24, section gap 16.
Sections from top to bottom; a flexible spacer pushes the nav bar to the bottom.

## Shared parts

- **Header row** (space-between, centred vertically): left = `bge-embed-rs` (title/600) over
  `Local bge-m3 embedding server` (caption, muted), gap 4. Right = **status pill**: surface bg,
  1 px border, full radius, padding 8×4, 8 px dot + label (caption/500). Dot colour and label by
  phase: Downloading → warning, Loading → warning, Ready → success, Failed → error.
- **Card**: surface bg, 1 px border, radius 8, padding 16, fills the width.
- **Primary button**: accent bg, radius 4, padding 16×8, label caption/600 in `accent.on`.
- **Secondary button**: surface bg + 1 px border, radius 4, padding 16×8, label caption/500.
- **Nav bar** (bottom): card-style bar, padding 4, three equal tabs (Status / Connections /
  Settings), tab padding 8, radius 4. Active tab = accent bg + `accent.on` label (600);
  inactive = no bg + muted label (500).

## Screen: Status

1. Header row.
2. **Model card**: row `Model · BAAI/bge-m3` (body/600) — right: `62%` while downloading,
   `Loaded` when ready (body/500). Progress track 8 px high, radius 4, border colour; fill =
   accent. Detail line (caption, muted): `Downloading pytorch_model.bin · 1.41 / 2.27 GB`, or
   `In memory · 1024-dim vectors · CPU (AVX2)` when ready. Loading phase: indeterminate bar.
3. **Endpoint card**: `Endpoint` (body/600). Row: URL field (window bg, radius 4, padding 8,
   caption/500 muted text, fills width) + primary `Copy` button, gap 8. Hint (caption, muted,
   wraps): `Available once the model is loaded. OpenAI-compatible, model name: bge-m3` before
   ready, `OpenAI-compatible · model name: bge-m3` after. Copy disabled until Ready.
4. **Stats row**: three equal cards, gap 12, each: label (caption, muted) over value
   (subtitle/600), gap 4 — `Requests`, `Texts embedded`, `Last latency` (`— ms` until first request).
5. Spacer, nav bar.

Failed phase: model card shows the error message in `status.error`, pill = Failed.

## Screen: Connections

1. Header row.
2. **Card "Detected on this computer"** (title body/600, row gap 12). One row per detected tool:
   32 px circle avatar (window bg + border, 2-letter initials caption/600 muted) · info column
   (tool name body/600 over meta row: 6 px status dot + caption muted text, gap 4) · button.
   - connected → success dot, `Connected · localhost:5055`, secondary `Re-check`
   - needs a key → warning dot, `Needs an admin API key · localhost:8080`, primary `Connect`
   - found → muted dot, `Found · localhost:3001`, primary `Connect`
3. **Card "Other tools (manual setup)"**: same rows, muted dot, secondary `Copy config`
   (copies ready-to-paste settings).
4. Note (caption, muted, wraps): `Tools running in Docker reach this server at
   host.docker.internal:11435. Enable network access in Settings on Linux.`
5. Spacer, nav bar.

## Screen: Settings

1. Header row.
2. **Card "Server"**: setting rows (label body/500 over hint caption muted, wraps; control on
   the right, gap 12):
   - `Port` — hint `The address tools connect to. Restart required.` — input 88×32 (window bg,
     border, radius 4, padding 8, body text).
   - `Parallel threads` — hint `More threads = faster, but uses more CPU. 4 is a good default.` — input.
3. **Card "Access & startup"**:
   - `Allow network and Docker access` — hint `Listens on all interfaces. No password - only
     enable on trusted networks.` — toggle (off).
   - `Start when I log in` — toggle (on).
   Toggle: 36×20, full radius, padding 4, 12 px knob in surface colour; track = accent when on,
   `text.muted` when off; knob right when on, left when off.
4. Spacer, nav bar.
