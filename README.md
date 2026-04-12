# Codesoft Internship Projects

A collection of three Python projects developed during the Codesoft Internship, covering foundational to advanced concepts in AI, machine learning, and algorithm design.

---

## Projects

| # | Project | Type | Complexity |
|---|---------|------|------------|
| 1 | [AI Chatbot](#1-ai-chatbot) | Rule-Based NLP | Beginner |
| 2 | [Movie Recommendation System](#2-movie-recommendation-system) | Machine Learning | Intermediate |
| 3 | [Tic-Tac-Toe AI](#3-tic-tac-toe-ai) | Game AI / Algorithms | Advanced |

---

## 1. AI Chatbot

A rule-based conversational assistant built with Python's standard library. Uses regular expressions to match user input and return contextually appropriate responses — no external dependencies required.

### Features

- Greeting recognition (`hi`, `hello`, `hey`, `hola`, and more)
- Current date and time on demand
- Help menu listing all supported commands
- Graceful fallback for unrecognised input
- Case-insensitive, whitespace-tolerant input handling

### Installation & Usage

```bash
cd "Ai chat bot"
python chatbot.py
```

**Example session:**

```
Rule-Based Chatbot
Type 'bye' to end the chat.

You: hello
Bot: Hello! How can I help you today?

You: what's the time?
Bot: Current time is 03:45 PM.

You: thank you
Bot: You're welcome!

You: bye
Bot: Goodbye! Have a nice day.
```

### Supported Commands

| Input | Response |
|-------|----------|
| `hi`, `hello`, `hey`, `hola` | Greeting |
| `who are you?`, `name` | Self-introduction |
| `help`, `what can you do?` | Lists all features |
| `what time?`, `current time` | Current time (HH:MM AM/PM) |
| `what's today?`, `date` | Today's date (DD Month YYYY) |
| `how are you?` | Status response |
| `thanks`, `thank you` | Acknowledgement |
| `bye`, `exit`, `quit` | Ends session |
| *(anything else)* | Friendly fallback message |

### How It Works

```
User Input
    ↓
Normalise (lowercase + strip whitespace)
    ↓
Pattern Matching (regex)
    ↓
Return Response
    ↓
Continue or Exit
```

Key regex concepts used: `\b` (word boundaries), `|` (alternation), `()` (grouping).

### Project Structure

```
Ai chat bot/
├── chatbot.py      # Chatbot logic (~60 lines)
└── README.md
```

**Stats:** ~60 lines of code · 2 functions · 8 regex patterns · 0 dependencies

### Limitations & Future Improvements

The chatbot is intentionally minimal — it has no memory of previous messages, no semantic understanding, and returns static responses. Planned extensions include NLP integration (spaCy), context awareness, a web interface (Flask), sentiment analysis, and API-connected features (weather, news).

---

## 2. Movie Recommendation System

A recommendation engine implementing two standard machine learning filtering techniques: **Content-Based Filtering** and **Collaborative Filtering**. Built with pandas, NumPy, and scikit-learn.

### Installation

```bash
pip install pandas numpy scikit-learn
# or
pip install -r requirements.txt
```

### Quick Start

```bash
cd "RECOMMENDATION SYSTEM"
python main.py
```

**Expected output:**

```
==================================================
1. CONTENT-BASED FILTERING
==================================================
Movies similar to 'The Matrix':
  1. Inception
  2. Interstellar
  3. Avengers: Endgame

==================================================
2. COLLABORATIVE FILTERING (USER-BASED)
==================================================
Recommendations for user 'u1':
  1. Avengers: Endgame
  2. Spider-Man: No Way Home
```

### How It Works

#### Content-Based Filtering

Recommends movies by comparing genre feature vectors using cosine similarity.

```
Similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

1. Extract genre strings from each movie
2. Vectorise using `CountVectorizer`
3. Build an n×n cosine similarity matrix
4. Return the top-N most similar movies to the query title

Best suited for new users with no rating history (cold-start solution).

#### Collaborative Filtering (User-Based)

Recommends movies by finding users with similar rating patterns.

1. Build a User × Movie rating matrix
2. Compute cosine similarity between all user vectors
3. Identify the most similar peer to the target user
4. Recommend highly rated (>3★) movies the target user has not yet seen

Best suited for returning users with an established rating history.

#### Comparison

| Aspect | Content-Based | Collaborative |
|--------|--------------|---------------|
| Data required | Item features only | User ratings |
| Cold-start | ✅ Handles new users | ❌ Requires history |
| Diversity | Limited (genre-bound) | Higher (cross-genre) |
| Scalability | O(n×m²) | O(u²×m) |
| Novelty | Predictable | Serendipitous |

### API Reference

```python
from main import get_content_based_recommendations, get_collaborative_recommendations

# Content-based
get_content_based_recommendations(movie_title="The Matrix", top_n=3)
# → ['Inception', 'Interstellar', 'Avengers: Endgame']

# Collaborative
get_collaborative_recommendations(target_user="u1", top_n=2)
# → ['Avengers: Endgame', 'Spider-Man: No Way Home']

# Hybrid (manual combination)
content = get_content_based_recommendations("Inception", top_n=2)
collab  = get_collaborative_recommendations("u1", top_n=2)
hybrid  = list(set(content + collab))
```

### Dataset

**Movies**

| ID | Title | Genres |
|----|-------|--------|
| 1 | Inception | Action · Sci-Fi · Thriller |
| 2 | The Matrix | Action · Sci-Fi |
| 3 | Interstellar | Adventure · Drama · Sci-Fi |
| 4 | The Notebook | Drama · Romance |
| 5 | Titanic | Drama · Romance |
| 6 | Avengers: Endgame | Action · Adventure · Sci-Fi |
| 7 | Spider-Man: No Way Home | Action · Adventure · Fantasy |

**User Profiles**

| User | Preference | Example Ratings |
|------|-----------|-----------------|
| u1 | Sci-Fi / Action | Inception ★5, The Matrix ★4, Interstellar ★5 |
| u2 | Romance / Drama | The Notebook ★5, Titanic ★5, Interstellar ★2 |
| u3 | Sci-Fi / Action | Similar to u1 |
| u4 | Mixed | Varied |

### Extending the System

```python
# Add a movie
movies_data['movie_id'].append(8)
movies_data['title'].append('The Dark Knight')
movies_data['genres'].append('Action Crime Thriller')

# Load a real dataset (e.g. MovieLens)
ratings_df = pd.read_csv('ratings.csv')
movies_df  = pd.read_csv('movies.csv')

# Expose via Flask
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/recommendations/<user_id>')
def recommend(user_id):
    return jsonify({'recommendations': get_collaborative_recommendations(user_id)})
```

### Project Structure

```
RECOMMENDATION SYSTEM/
├── main.py             # Recommendation engine
├── requirements.txt    # Dependencies
└── README.md
```

**Dependencies:** `pandas ≥ 2.0`, `numpy ≥ 1.24`, `scikit-learn ≥ 1.2`

---

## 3. Tic-Tac-Toe AI

A command-line Tic-Tac-Toe game featuring an **unbeatable AI opponent** powered by the Minimax algorithm with Alpha-Beta Pruning. Pure Python — no external libraries required.

### Installation & Usage

```bash
cd TIC-TAC-TOE
python tictactoe.py
```

Board positions are numbered 0–8:

```
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8
```

You play as **X**; the AI plays as **O**. On each turn, enter the position number where you want to place your mark. The game ends when a player completes a row, column, or diagonal — or when all nine squares are filled (draw).

### Algorithms

#### Minimax

Minimax exhaustively explores all possible future game states, scoring each terminal position and propagating scores back to the root to identify the move with the optimal guaranteed outcome.

**Scoring:** AI win → `+10 − depth` (prefer faster wins) · Human win → `−10 + depth` (delay losses) · Draw → `0`

```
minimax(board, depth, isMaximising, α, β):
    if terminal state → return score
    if isMaximising (AI):
        best = -∞
        for each move:
            best = max(best, minimax(next_state, depth+1, false, α, β))
            α = max(α, best)
            if β ≤ α: break  ← prune
        return best
    else (Human):
        best = +∞
        for each move:
            best = min(best, minimax(next_state, depth+1, true, α, β))
            β = min(β, best)
            if β ≤ α: break  ← prune
        return best
```

#### Alpha-Beta Pruning

Alpha-Beta extends Minimax by tracking the best scores found so far for each player (`α` for the maximiser, `β` for the minimiser) and skipping any branch that cannot possibly influence the final decision.

| Metric | Value |
|--------|-------|
| Nodes without pruning | 362,880 (9!) |
| Nodes with pruning | ~1,000 |
| Reduction | ~99.7% |
| First-move latency | ~140 ms |

#### Why the AI Is Unbeatable

Minimax with perfect information evaluates every reachable game state. Given optimal play, the best a human opponent can achieve is a draw — the AI will never lose.

### Performance

| Game Phase | Nodes Evaluated | Latency |
|------------|----------------|---------|
| Opening move | ~1,000 | ~140 ms |
| Mid-game | ~100 | ~10 ms |
| End-game | ~10 | ~1 ms |

### Game Strategy

**Position priority:** Center (4) > Corners (0, 2, 6, 8) > Edges (1, 3, 5, 7)

**AI logic:** Win immediately if possible → block an imminent human win → play the highest-value available position.

**Tip for humans:** Taking the center on your first move and playing corner-to-corner is the best strategy for forcing a draw.

### Running Tests

```bash
python test_tictactoe.py
```

Five test categories: board logic, Minimax correctness, edge cases, performance benchmarks, and full game simulations.

### Project Structure

```
TIC-TAC-TOE/
├── tictactoe.py          # Game + AI implementation (~300 lines)
├── test_tictactoe.py     # Test suite (~350 lines)
├── ALGORITHM_GUIDE.md    # In-depth algorithm documentation
├── QUICKSTART.md         # 30-second setup guide
├── PROJECT_SUMMARY.md    # Project overview
└── README.md
```

**Stats:** ~650 lines of code · 0 dependencies · 8 winning combinations tracked

### Possible Extensions

- Difficulty levels (randomised moves, depth-limited search)
- GUI with Tkinter or Pygame
- Network multiplayer via sockets
- Monte Carlo Tree Search (MCTS)
- Neural network opponent (TensorFlow / PyTorch)
- Transposition tables for memoisation

---

## License

All projects are open source and freely available for learning and modification.

---

*Created for the Codesoft Internship · April 2026*
