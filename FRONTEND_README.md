# Code Smell Detection Frontend

A modern, responsive web UI for the Code Smell Detection API. Built with HTML5, Bootstrap 5, and vanilla JavaScript.

## Features

### 🎯 Code Analysis Interface
- **Language Selection**: Support for Java, Python, and JavaScript
- **Code Editor**: Large textarea with syntax highlighting support
- **File Metadata**: Optional filename input for better organization
- **Configuration Options**:
  - RAG Context Toggle: Enable/disable retrieval-augmented generation
  - Timeout Control: Set analysis timeout (30-600 seconds)
- **Character Count**: Real-time character counter

### 📊 Real-Time Progress Tracking
- **Progress Bar**: Visual progress indicator with percentage
- **Status Steps**: Shows current analysis phase:
  - Queued (5%)
  - Parsing (15%)
  - RAG Retrieval (35%)
  - Inference (65%)
  - Validation (85%)
  - Completed (100%)

### 📈 Results Display
- **Metrics Dashboard**: 
  - Code Smells Count
  - Maximum Severity Level
  - Analysis Time Duration
- **Findings List**: Detailed code smell detection results with:
  - Issue Name and Type
  - Severity Badge (High/Medium/Low)
  - Line Number/Location
  - Description and Explanation
  - AI-Powered Suggestions for fixes
- **Code Metrics**: Additional metrics from the analysis

### 🏥 System Health Dashboard
- **Service Status**: Real-time monitoring of:
  - Ollama LLM Service
  - ChromaDB Vector Database
  - Database Connection
- **System Metrics**:
  - Uptime Counter
  - Active Analyses Count
  - Completed Analyses Count
  - Cache Size (in MB)

### 📜 Recent Analyses Log
- **Analysis History**: Table showing last 10 analyses with:
  - Analysis ID (truncated)
  - Programming Language
  - Status Badge
  - Creation Timestamp
- **Auto-Refresh**: Updates every 10 seconds

### 🔔 Notifications
- **Toast Messages**: Non-blocking notifications for:
  - Successful submissions
  - Analysis completions
  - Error messages
  - Status updates

## Technical Stack

- **Frontend Framework**: Bootstrap 5.3.0
- **Icon Library**: Bootstrap Icons 1.11.0
- **JavaScript**: Vanilla ES6+ (no jQuery required)
- **HTTP Client**: Fetch API
- **Real-Time Updates**: Poll-based progress tracking (1s intervals)

## File Structure

```
/src/static/
├── index.html          # Main UI structure and layout
├── style.css           # Bootstrap customizations and animations
└── app.js              # API integration and interactions
```

## API Integration

The frontend communicates with the FastAPI backend using the following endpoints:

### Analysis Endpoints
- **POST** `/api/v1/analyze` - Submit code for analysis
  - Returns: `202 Accepted` with `analysis_id`
- **GET** `/api/v1/progress/{analysis_id}` - Get progress (0-100%)
  - Returns: `percentage`, `status` (queued/processing/completed/failed)
- **GET** `/api/v1/results/{analysis_id}` - Get analysis results
  - Returns: `findings`, `metrics`, `duration`, `analysis_id`
- **GET** `/api/v1/analyses/active` - Get recent analyses list
  - Returns: List of up to 10 recent analyses

### Health Endpoints
- **GET** `/api/v1/status` - System status and metrics
  - Returns: `uptime_seconds`, `active_analyses`, `completed_analyses`, `cache_size_bytes`, `health_checks`

## Usage Instructions

### 1. Starting the Server

```bash
cd /Users/bibekgupta/Downloads/projects/code-smell
source .venv/bin/activate
python3 -m uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000
```

### 2. Accessing the Frontend

Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Submitting Code for Analysis

1. Select a **Programming Language** (Java, Python, or JavaScript)
2. (Optional) Enter a **File Name** for reference
3. Paste your code snippet in the **Code Editor**
4. Configure analysis options:
   - Enable/disable **RAG Context**
   - Set **Analysis Timeout** (default: 300s)
5. Click **Analyze Code** button
6. Monitor progress in the **Results** tab
7. View detailed findings when complete

### 4. Understanding Results

#### Finding Severity Levels
- **High (Red)**: Critical code quality issues that should be fixed immediately
- **Medium (Yellow)**: Important issues that should be addressed
- **Low (Blue)**: Minor suggestions for code improvement

#### Suggested Actions
- Each finding includes a **Suggestion** box with recommended fixes
- Implement suggestions to improve code quality

### 5. Keyboard Shortcuts

- **Ctrl+Enter** or **Cmd+Enter**: Quick submit from code editor

## UI Components

### Theme and Colors
- **Primary**: Blue (#3b82f6)
- **Secondary**: Purple (#8b5cf6)
- **Success**: Green (#10b981)
- **Danger**: Red (#ef4444)
- **Warning**: Amber (#f59e0b)

### Responsive Design
- **Desktop** (>992px): 3-column layout (sidebar + content)
- **Tablet** (768-992px): 2-column responsive layout
- **Mobile** (<768px): Single-column stack layout

## Performance Optimizations

1. **Auto-Refresh Intervals**:
   - Status dashboard: 5 seconds
   - Recent analyses: 10 seconds
   - Progress polling: 1 second (during analysis)

2. **Resource Management**:
   - Clears progress intervals when analysis completes
   - Lazy-loads analysis history
   - Efficient DOM updates

3. **Error Handling**:
   - Network error recovery
   - Graceful fallback messages
   - User-friendly error notifications

## Browser Compatibility

- Chrome/Chromium: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Edge: ✅ Full support

## Development Tips

### Debugging
- Open **Browser DevTools** (F12) for console logs
- Check **Network tab** to monitor API calls
- Frontend logs all API requests and responses

### API Testing
- Use the **API Documentation** at `/docs` (Swagger UI)
- Test endpoints before UI integration
- Monitor response times in browser DevTools

### Customization
- Edit `style.css` to change colors or layout
- Modify `app.js` to add new features
- Update `index.html` to change UI structure

## Known Limitations

1. **In-Memory State**: Analysis results stored in-memory on server (not persistent)
2. **Frontend State**: Form doesn't persist across page refresh
3. **Browser Cache**: Results may be cached by browser

## Future Enhancements

- [ ] Dark mode toggle
- [ ] Export results to PDF
- [ ] Comparison between multiple analyses
- [ ] Code highlighting in editor
- [ ] WebSocket for real-time progress (instead of polling)
- [ ] Analysis history persistence
- [ ] User authentication
- [ ] Advanced filtering and search

## Troubleshooting

### "API connection failed"
- Ensure backend server is running on port 8000
- Check browser console for CORS errors
- Verify API endpoints are accessible

### "Analysis times out"
- Increase timeout value in the form
- Reduce code snippet size
- Check server resources

### "Results not displaying"
- Refresh the page
- Clear browser cache
- Check browser console for errors

## Support

For issues or questions:
1. Check the browser console (F12) for errors
2. Review API documentation at `/docs`
3. Check FastAPI server logs for backend errors
