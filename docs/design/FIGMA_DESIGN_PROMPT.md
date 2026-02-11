# Figma Design Prompt
## UI/UX Design for LLM-Based Code Review System

**Project:** Code Smell Detection & Analysis Tool  
**Platform:** Web Application (Responsive)  
**Design System:** Component-Based, Multi-Screen  
**Last Updated:** February 9, 2026

---

## Design Brief

Create a professional, developer-focused web application design for an LLM-based code review system that detects and explains code smells. The design should emphasize clarity, professionalism, and ease of use while showcasing AI-powered analysis capabilities.

---

## 1. Design System Specifications

### 1.1 Color Palette (Industry Standard - Developer Tools)

**Primary Colors:**
```
Primary Blue (Main Actions)
- Primary 500: #0066CC (Buttons, Links)
- Primary 400: #3385D6 (Hover states)
- Primary 600: #0052A3 (Active states)
- Primary 100: #E6F2FF (Backgrounds)

Secondary Gray (Text & Backgrounds)
- Gray 900: #1A1A1A (Primary text)
- Gray 700: #4A4A4A (Secondary text)
- Gray 500: #A0A0A0 (Disabled text)
- Gray 300: #D1D1D1 (Borders)
- Gray 100: #F5F5F5 (Light backgrounds)
- Gray 50: #FAFAFA (Subtle backgrounds)

Success Green (Positive Indicators)
- Success 500: #10B981 (Success states)
- Success 100: #D1FAE5 (Success backgrounds)

Warning Yellow (Code Smell Warnings)
- Warning 500: #F59E0B (Warning states)
- Warning 100: #FEF3C7 (Warning backgrounds)

Error Red (Critical Issues)
- Error 500: #EF4444 (Error states)
- Error 100: #FEE2E2 (Error backgrounds)

Accent Purple (AI/LLM Features)
- Accent 500: #8B5CF6 (AI highlights)
- Accent 100: #EDE9FE (AI backgrounds)
```

**Usage Guide:**
- **Primary Blue:** Main CTAs, active states, links
- **Gray Scale:** Text hierarchy, backgrounds, dividers
- **Success Green:** Successful operations, "no issues found"
- **Warning Yellow:** Medium severity code smells
- **Error Red:** High severity issues, errors
- **Accent Purple:** LLM-powered features, AI insights

### 1.2 Typography

**Font Family:**
```
Headings: Inter (Sans-serif) - Weight: 600, 700
Body Text: Inter (Sans-serif) - Weight: 400, 500
Code: JetBrains Mono (Monospace) - Weight: 400, 500
```

**Type Scale:**
```
Hero/H1: 48px / 3rem - Line height: 1.2 - Weight: 700
H2: 36px / 2.25rem - Line height: 1.3 - Weight: 600
H3: 28px / 1.75rem - Line height: 1.4 - Weight: 600
H4: 24px / 1.5rem - Line height: 1.4 - Weight: 600
H5: 20px / 1.25rem - Line height: 1.5 - Weight: 500
Body Large: 18px / 1.125rem - Line height: 1.6 - Weight: 400
Body: 16px / 1rem - Line height: 1.6 - Weight: 400
Body Small: 14px / 0.875rem - Line height: 1.5 - Weight: 400
Caption: 12px / 0.75rem - Line height: 1.4 - Weight: 400
Code: 14px / 0.875rem - Line height: 1.6 - Weight: 400 (JetBrains Mono)
```

### 1.3 Spacing System (8px Grid)

```
XXS: 4px (0.25rem)
XS: 8px (0.5rem)
SM: 12px (0.75rem)
MD: 16px (1rem)
LG: 24px (1.5rem)
XL: 32px (2rem)
2XL: 48px (3rem)
3XL: 64px (4rem)
4XL: 96px (6rem)
```

### 1.4 Component Specifications

**Buttons:**
```
Primary Button:
- Height: 40px (MD), 48px (LG)
- Padding: 12px 24px (MD), 14px 28px (LG)
- Border Radius: 8px
- Font Weight: 500
- Background: Primary 500
- Text: White
- Hover: Primary 400
- Active: Primary 600

Secondary Button:
- Same dimensions as primary
- Background: Gray 100
- Text: Gray 900
- Border: 1px solid Gray 300
- Hover: Gray 200

Tertiary/Link Button:
- No background
- Text: Primary 500
- Underline on hover
```

**Input Fields:**
```
Text Input:
- Height: 40px
- Padding: 10px 12px
- Border: 1px solid Gray 300
- Border Radius: 6px
- Focus: Border Primary 500, Shadow
- Error: Border Error 500

Textarea (Code Input):
- Min Height: 200px
- Max Height: 600px
- Font: JetBrains Mono 14px
- Padding: 12px
- Line Numbers: Optional
```

**Cards:**
```
Card Container:
- Background: White
- Border: 1px solid Gray 200
- Border Radius: 12px
- Padding: 24px
- Shadow: 0 1px 3px rgba(0,0,0,0.1)

Code Smell Card:
- Border-left: 4px solid (severity color)
- Warning: border-left Warning 500
- Error: border-left Error 500
- Success: border-left Success 500
```

**Badges:**
```
Severity Badges:
- High: Error 500 background, white text
- Medium: Warning 500 background, white text
- Low: Success 500 background, white text
- Padding: 4px 12px
- Border Radius: 12px (pill shape)
- Font Size: 12px
- Font Weight: 500
```

---

## 2. Screen Designs Required

### 2.1 Dashboard / Home Screen

**Purpose:** Main entry point for code analysis

**Components:**
1. **Header Navigation**
   - Logo/Brand (left)
   - Navigation: Dashboard | History | Comparison | Settings
   - User profile/menu (right)

2. **Hero Section**
   - Headline: "AI-Powered Code Review"
   - Subheadline: "Detect code smells instantly with LLM analysis"
   - Primary CTA: "Analyze Code"

3. **Quick Stats Cards** (3-column grid)
   - Total Analyses (number + icon)
   - Smells Detected (number + icon)
   - Average Code Quality (score + progress ring)

4. **Code Input Section**
   - Language selector dropdown (Java, Python, etc.)
   - Large textarea with syntax highlighting
   - File upload option (drag & drop zone)
   - Analysis mode selector (Quick/Thorough/Comprehensive)
   - "Analyze Code" button (primary, large)

5. **Recent Analyses** (List)
   - Timestamp, Language, Smells count
   - Quick view / Delete actions

**Layout:**
- Desktop: 1200px max-width, centered
- Tablet: Responsive 2-column for stats
- Mobile: Single column, stacked

**Figma Prompt:**
```
Design a modern developer dashboard for a code review tool. Include:
- Clean header with navigation (Dashboard, History, Comparison, Settings)
- Hero section with headline "AI-Powered Code Review" and CTA button
- 3 stat cards showing metrics (Total Analyses, Smells Detected, Code Quality)
- Large code input area with syntax highlighting, language selector dropdown
- File upload drag-and-drop zone
- Analysis mode toggle (Quick/Thorough/Comprehensive)
- Recent analyses list with timestamps
- Use professional blue (#0066CC) and gray color scheme
- Inter font for UI, JetBrains Mono for code
- 1200px max-width, centered layout
- Include proper spacing and visual hierarchy
```

---

### 2.2 Analysis Results Screen

**Purpose:** Display detected code smells with AI explanations

**Components:**
1. **Results Header**
   - Back button
   - Overall assessment summary
   - Code quality score (0-100, circular progress)
   - Timestamp, Model used

2. **Filters & Sorting**
   - Filter by severity (All, High, Medium, Low)
   - Sort by (Severity, Location, Type)
   - Search smells

3. **Code Smell Cards** (Vertical list)
   Each card contains:
   - Smell type badge (colored)
   - Severity indicator (High/Medium/Low)
   - Location in code (line numbers, method name)
   - Code snippet (with syntax highlighting, lines referenced)
   - AI Explanation (collapsible)
   - Refactoring suggestion (collapsible)
   - Confidence score (0-100%)

4. **Code Preview Panel** (Split view - optional)
   - Full code with line numbers
   - Highlighted problematic sections
   - Click smell card to scroll to location

5. **Action Bar**
   - Export report (PDF/JSON)
   - Share results
   - Compare with baseline tools

**Layout:**
- Desktop: Split view (results list + code preview)
- Tablet: Toggleable view
- Mobile: Stacked, expandable cards

**Figma Prompt:**
```
Design a code analysis results page with:
- Header showing overall assessment, code quality score (circular progress), timestamp
- Filter buttons for severity (All, High, Medium, Low)
- Vertical list of code smell cards, each with:
  - Colored severity badge (red=high, yellow=medium, green=low)
  - Smell type (e.g., "Long Method")
  - Location (line numbers)
  - Code snippet with syntax highlighting
  - Collapsible AI explanation section
  - Refactoring suggestion
  - Confidence percentage
- Optional split-view with full code preview on right side
- Export and share buttons in action bar
- Use color coding: red (#EF4444) for high severity, yellow (#F59E0B) medium, green (#10B981) low
- Card style: white background, 12px border-radius, 4px colored left border
```

---

### 2.3 Comparison View Screen

**Purpose:** Side-by-side comparison of LLM vs. baseline tools

**Components:**
1. **Comparison Header**
   - Title: "LLM vs. Baseline Tools"
   - Select tools to compare (checkboxes: SonarQube, PMD, LLM)

2. **Metrics Comparison** (Table/Cards)
   - Precision, Recall, F1-Score
   - Total smells detected
   - Execution time
   - Visual comparison (bar charts)

3. **Venn Diagram / Overlap Visualization**
   - Smells detected by both
   - Unique to LLM
   - Unique to baseline tool

4. **Detailed Comparison Table**
   Columns:
   - Smell Type
   - LLM Detection (✓/✗)
   - SonarQube Detection (✓/✗)
   - PMD Detection (✓/✗)
   - Agreement indicator

5. **False Positive Analysis**
   - List of false positives from each tool
   - Reasoning for disagreements

**Layout:**
- Desktop: Side-by-side comparison panels
- Tablet: Stacked with toggle
- Mobile: Swipeable comparison cards

**Figma Prompt:**
```
Design a comparison screen showing LLM vs. baseline tools:
- Header with tool selection checkboxes (SonarQube, PMD, LLM)
- Metrics comparison cards showing Precision, Recall, F1-Score for each tool
- Venn diagram showing overlap of detected smells
- Detailed comparison table with columns: Smell Type, LLM (✓/✗), SonarQube (✓/✗), PMD (✓/✗)
- Bar charts comparing metrics side-by-side
- False positive analysis section
- Use purple accent (#8B5CF6) for LLM-specific features
- Professional data visualization style
- Clear visual hierarchy for comparing results
```

---

### 2.4 History / Past Analyses Screen

**Purpose:** View and manage historical code analyses

**Components:**
1. **Search & Filters**
   - Search by code snippet, filename
   - Filter by date range
   - Filter by language
   - Filter by smell types

2. **Analysis Grid/List**
   Each item:
   - Thumbnail (code icon or first few lines)
   - Timestamp
   - Language badge
   - Smells count
   - Quality score
   - Actions: View, Delete, Compare

3. **Bulk Actions**
   - Select multiple
   - Delete selected
   - Export selected

4. **Pagination**
   - Show X of Y results
   - Page numbers or infinite scroll

**Layout:**
- Desktop: Grid (2-3 columns)
- Tablet: 2 columns
- Mobile: Single column list

**Figma Prompt:**
```
Design a history/archive page for past code analyses:
- Top section with search bar and filters (Date range, Language, Smell types)
- Grid layout of analysis cards (2-3 columns)
- Each card shows:
  - Code icon or preview
  - Timestamp
  - Language badge (colored)
  - Number of smells detected
  - Quality score (small circular progress)
  - Action buttons: View, Delete, Compare
- Checkboxes for bulk selection
- Bulk action buttons (Delete, Export)
- Pagination controls at bottom
- Clean, organized layout with proper card spacing
```

---

### 2.5 Settings / Configuration Screen

**Purpose:** Customize analysis preferences

**Components:**
1. **Settings Navigation** (Sidebar)
   - General
   - Analysis Preferences
   - Model Selection
   - Notifications
   - About

2. **General Settings**
   - Theme toggle (Light/Dark)
   - Language preference
   - Auto-save results

3. **Analysis Preferences**
   - Default analysis mode
   - Smell types to check (checkboxes)
   - Confidence threshold slider
   - Enable/disable specific detections

4. **Model Selection**
   - LLM model dropdown (Llama 3, CodeLlama, Mistral)
   - Temperature slider
   - Max tokens input

5. **About Section**
   - Version info
   - Model info
   - Dataset info
   - Links to documentation

**Layout:**
- Desktop: Sidebar + content area
- Tablet/Mobile: Stacked sections

**Figma Prompt:**
```
Design a settings page with:
- Left sidebar navigation: General, Analysis Preferences, Model Selection, About
- General settings section with theme toggle, language selector
- Analysis preferences with:
  - Default mode selector (Quick/Thorough/Comprehensive)
  - Smell types checklist
  - Confidence threshold slider (0-100%)
- Model selection dropdown (Llama 3, CodeLlama, Mistral)
- Temperature slider with label
- About section with version info, model details
- Toggle switches, sliders, dropdowns styled consistently
- Clean, organized form layout
```

---

### 2.6 Code Upload / File Selection Screen (Modal/Overlay)

**Purpose:** Upload code files for analysis

**Components:**
1. **Drag & Drop Zone**
   - Dashed border
   - Upload icon
   - "Drag & drop or click to browse"
   - Supported formats: .java, .py, .js, .txt

2. **File List**
   - Filename
   - File size
   - Remove button

3. **Batch Settings**
   - Apply same language to all
   - Analysis mode for batch

4. **Submit Button**
   - "Analyze X files"

**Figma Prompt:**
```
Design a file upload modal/overlay:
- Large drag-and-drop zone with dashed border, upload icon
- Text: "Drag & drop files or click to browse"
- Supported formats shown: .java, .py, .js
- File list below showing uploaded files with:
  - File icon
  - Filename
  - File size
  - Remove (X) button
- Batch settings: Language selector, Analysis mode
- Primary "Analyze X files" button
- Clean, focused modal design
```

---

## 3. Responsive Design Breakpoints

```
Desktop (Large): 1200px+
Desktop (Standard): 1024px - 1199px
Tablet (Landscape): 768px - 1023px
Tablet (Portrait): 600px - 767px
Mobile (Large): 480px - 599px
Mobile (Standard): 320px - 479px
```

**Responsive Adjustments:**
- Desktop: Full multi-column layouts, split views
- Tablet: 2-column grids, collapsible sidebars
- Mobile: Single column, stacked cards, hamburger menu

---

## 4. Component Library to Include

### 4.1 Core Components
- [ ] Buttons (Primary, Secondary, Tertiary, Icon)
- [ ] Input fields (Text, Textarea, Code editor)
- [ ] Dropdowns/Selects
- [ ] Checkboxes, Radio buttons, Toggle switches
- [ ] Sliders
- [ ] Progress indicators (Circular, Linear)
- [ ] Badges/Tags
- [ ] Cards (Default, Code smell, Stats)
- [ ] Modals/Overlays
- [ ] Tooltips
- [ ] Alerts/Notifications (Success, Warning, Error, Info)

### 4.2 Navigation Components
- [ ] Header/Top navigation
- [ ] Sidebar navigation
- [ ] Breadcrumbs
- [ ] Tabs
- [ ] Pagination

### 4.3 Data Display Components
- [ ] Tables (Comparison, Results)
- [ ] Code blocks (with syntax highlighting)
- [ ] Charts (Bar, Line, Pie, Venn diagram)
- [ ] Progress rings/scores
- [ ] Empty states
- [ ] Loading states (Skeleton screens, Spinners)

### 4.4 Specialized Components
- [ ] Code smell card
- [ ] Analysis result card
- [ ] File upload zone
- [ ] Code preview panel
- [ ] Severity indicators
- [ ] Confidence meters

---

## 5. Iconography

**Icon Style:** Outline style (2px stroke), 24x24px base size

**Icons Needed:**
- Code/Document (code analysis)
- Upload (file upload)
- Download (export)
- Search (search functionality)
- Filter (filtering)
- Settings (configuration)
- User (profile)
- Dashboard (home)
- History (clock/calendar)  
- Compare (arrows/split)
- Check/Success (checkmark)
- Warning (exclamation triangle)
- Error (X in circle)
- Info (i in circle)
- AI/Brain (LLM features)
- Refresh (reload)
- Delete/Trash (remove)
- Edit/Pen (modify)
- Share (share results)
- Menu (hamburger)

**Recommended Icon Library:** Heroicons, Lucide Icons, or Phosphor Icons

---

## 6. Interaction States

**Buttons:**
- Default
- Hover (lighten 10%)
- Active/Pressed (darken 10%)
- Disabled (opacity 50%, cursor not-allowed)
- Loading (spinner + disabled)

**Input Fields:**
- Default
- Focus (border highlight, subtle shadow)
- Error (red border, error icon)
- Disabled (grayed out)
- Filled

**Cards:**
- Default
- Hover (subtle elevation increase)
- Selected (border highlight)
- Active/Expanded

---

## 7. Accessibility Considerations

- **Color Contrast:** WCAG AA compliant (4.5:1 for text)
- **Focus States:** Visible focus indicators on all interactive elements
- **Keyboard Navigation:** All functions accessible via keyboard
- **Screen Reader Support:** Proper ARIA labels, alt text
- **Font Sizes:** Minimum 14px for body text, scalable
- **Touch Targets:** Minimum 44x44px for mobile

---

## 8. Design Deliverables Checklist

- [ ] Complete design system (colors, typography, spacing, components)
- [ ] Dashboard / Home screen (Desktop, Tablet, Mobile)
- [ ] Analysis Results screen (Desktop, Tablet, Mobile)
- [ ] Comparison View screen (Desktop, Tablet, Mobile)
- [ ] History screen (Desktop, Tablet, Mobile)
- [ ] Settings screen (Desktop, Tablet, Mobile)
- [ ] File Upload modal (Responsive)
- [ ] Component library (all reusable components)
- [ ] Icon set (24x24px, outline style)
- [ ] Interaction states for all components
- [ ] Loading states, empty states, error states
- [ ] Responsive breakpoint examples

---

## 9. Tools & Plugins Recommended

**Figma Plugins:**
- **Stark:** Accessibility contrast checking
- **Iconify:** Access to icon libraries
- **Content Reel:** Generate placeholder content
- **Unsplash:** Stock images
- **Auto Layout:** For responsive components
- **Component:** For organizing design system

**Design Tokens:**
- Export colors, typography, spacing as JSON for developers

---

## 10. Final Figma Prompt (Comprehensive)

```
Design a complete UI/UX system for an AI-powered code review web application that detects code smells using LLMs. 

**Style Requirements:**
- Professional, developer-focused aesthetic
- Color palette: Primary blue #0066CC, grays for text, semantic colors (green success, yellow warning, red error, purple for AI features)
- Typography: Inter for UI (weights 400, 500, 600, 700), JetBrains Mono for code
- Spacing: 8px grid system
- Border radius: 6-12px for components
- Shadows: Subtle, elevation-based

**Screens to Design (Desktop, Tablet, Mobile):**
1. Dashboard with code input, stats cards, recent analyses
2. Results page with code smell cards, explanations, confidence scores
3. Comparison view (LLM vs. SonarQube/PMD) with metrics, Venn diagram
4. History/archive page with searchable analysis grid
5. Settings page with preferences, model selection
6. File upload modal with drag-and-drop

**Components:**
- Buttons (primary, secondary, tertiary)
- Input fields, textareas, code editor
- Cards (default, code smell, stats)
- Badges for severity (high=red, medium=yellow, low=green)
- Progress indicators (circular scores, linear bars)
- Tables, charts, Venn diagrams
- Navigation header, sidebar
- Modals, tooltips, alerts

**Key Features:**
- Syntax-highlighted code blocks
- Severity color coding (red/yellow/green)
- Collapsible explanation sections
- Split-view code preview
- Drag-and-drop file upload
- Responsive across breakpoints (1200px+, 768px, 320px)

**Accessibility:**
- WCAG AA contrast ratios
- Clear focus states
- Minimum 44px touch targets

Create a cohesive, professional design system with all screens and components organized in Figma with proper auto-layout and variants.
```

---

**Document Version:** 1.0  
**For:** Design Team / Figma Designer  
**Project:** Code Review LLM System  
**Maintained By:** Product Design Lead
