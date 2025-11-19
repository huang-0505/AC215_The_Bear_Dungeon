# Campaign Integration Guide

## Starting a Pre-Defined Campaign (e.g., Dragons of Stormwreck Isle)

### Overview

The orchestrator now supports **pre-loaded campaigns** with rich narratives and structured game states. You can start campaigns like "Dragons of Stormwreck Isle" without manually creating the game tree - it's automatically initialized with campaign-specific content.

---

## How Campaign Loading Works

### 1. Campaign Templates ([campaign_loader.py](src/orchestrator/campaign_loader.py))

Pre-defined campaigns include:
- **Opening Narrative**: Rich, immersive story introduction
- **Starting Location**: Where the adventure begins
- **Initial Quest**: The first objective
- **Difficulty Level**: Beginner, Medium, Hard, Very Hard

### 2. Automatic Game Tree Creation

When you start a campaign:
1. Orchestrator creates a `GameStateTree` with a root narration node
2. Campaign metadata is stored in the root node
3. Initial narrative is generated using campaign template
4. Character details are woven into the story
5. Game is ready to play!

**You don't need to pre-build the tree** - it's created on-the-fly when the player starts.

---

## Available Campaigns

### 1. Dragons of Stormwreck Isle (Beginner)
```json
{
  "id": "stormwreck-isle",
  "name": "Dragons of Stormwreck Isle",
  "difficulty": "beginner",
  "description": "A beginner-friendly adventure on a mysterious island inhabited by dragons."
}
```

### 2. Lost Mine of Phandelver (Medium)
```json
{
  "id": "classic-dungeon",
  "name": "The Lost Mine of Phandelver",
  "difficulty": "medium"
}
```

### 3. Tomb of Annihilation (Hard)
```json
{
  "id": "wilderness-adventure",
  "name": "Tomb of Annihilation",
  "difficulty": "hard"
}
```

### 4. Curse of Strahd (Hard)
```json
{
  "id": "gothic-horror",
  "name": "Curse of Strahd",
  "difficulty": "hard"
}
```

### 5. Descent into Avernus (Very Hard)
```json
{
  "id": "planar-adventure",
  "name": "Planescape: Descent into Avernus",
  "difficulty": "very hard"
}
```

---

## API Usage

### List All Campaigns

```bash
GET /campaigns
```

**Response**:
```json
{
  "campaigns": [
    {
      "id": "stormwreck-isle",
      "name": "Dragons of Stormwreck Isle",
      "description": "A beginner-friendly adventure...",
      "difficulty": "beginner"
    },
    ...
  ]
}
```

### Get Campaign Details

```bash
GET /campaigns/stormwreck-isle
```

**Response**:
```json
{
  "id": "stormwreck-isle",
  "name": "Dragons of Stormwreck Isle",
  "description": "A beginner-friendly adventure...",
  "opening_narrative": "You stand on the deck of the merchant vessel...",
  "starting_location": "Dragon's Rest Harbor",
  "initial_quest": "Explore Stormwreck Isle...",
  "difficulty": "beginner"
}
```

### Start a Campaign

```bash
POST /game/start
Content-Type: application/json

{
  "campaign_id": "stormwreck-isle",
  "character_class": "Fighter",
  "character_name": "Thorin Ironforge"
}
```

**Response**:
```json
{
  "session_id": "abc-123-def",
  "response": "You are Thorin Ironforge, a battle-hardened warrior...\n\nYou stand on the deck...",
  "campaign_info": {
    "campaign_id": "stormwreck-isle",
    "campaign_name": "Dragons of Stormwreck Isle",
    "starting_location": "Dragon's Rest Harbor",
    "initial_quest": "Explore Stormwreck Isle...",
    "difficulty": "beginner",
    "character_class": "Fighter",
    "character_name": "Thorin Ironforge"
  },
  "state": { ... }
}
```

---

## Frontend Integration

### Option 1: Update Existing Frontend (src/frontend)

Your current [game/page.tsx](src/frontend/app/game/page.tsx) can be enhanced:

```typescript
// Add campaign selection before starting game

const [campaigns, setCampaigns] = useState([]);
const [selectedCampaign, setSelectedCampaign] = useState(null);

// Fetch campaigns
useEffect(() => {
  fetch('/api/campaigns')
    .then(res => res.json())
    .then(data => setCampaigns(data.campaigns));
}, []);

// Start game with campaign
const startCampaignGame = async () => {
  const response = await fetch('/api/game/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      campaign_id: selectedCampaign,
      character_class: characterClass,
      character_name: characterName
    })
  });

  const data = await response.json();
  setSessionId(data.session_id);

  // Add opening narrative to messages
  setMessages([{
    author: 'ai',
    text: data.response,
    timestamp: Date.now()
  }]);
};
```

### Option 2: Use frontend-tmp for Testing

The [frontend-tmp](src/frontend-tmp) can be quickly modified:

```javascript
// In main.js, update the app structure

class CampaignApp {
  constructor() {
    this.sessionId = null;
    this.currentState = 'campaign-select'; // 'campaign-select', 'narration', 'combat'

    // Add campaign selector
    this.setupCampaignSelector();
  }

  async setupCampaignSelector() {
    const response = await fetch('http://localhost:8000/campaigns');
    const data = await response.json();

    // Display campaigns for selection
    this.displayCampaigns(data.campaigns);
  }

  async startCampaign(campaignId, characterClass, characterName) {
    const response = await fetch('http://localhost:8000/game/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        campaign_id: campaignId,
        character_class: characterClass,
        character_name: characterName
      })
    });

    const data = await response.json();
    this.sessionId = data.session_id;
    this.currentState = 'narration';

    // Display opening narrative
    this.addNarrative(data.response);
  }

  async sendAction(actionText) {
    const response = await fetch('http://localhost:8000/game/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: this.sessionId,
        text: actionText
      })
    });

    const data = await response.json();

    // Handle state transitions
    if (data.transition === 'narration -> combat') {
      this.currentState = 'combat';
      this.switchToCombatUI(data.combat_session_id);
    } else if (data.transition === 'combat -> narration') {
      this.currentState = 'narration';
      this.switchToNarrationUI();
    }

    // Display response
    this.addMessage(data.response);
  }

  switchToCombatUI(combatSessionId) {
    // Show combat interface from frontend-tmp/backend-tmp
    // Connect to combat agent at localhost:9000
    this.combatInterface = new CombatApp();
    this.combatInterface.sessionId = combatSessionId;
  }

  switchToNarrationUI() {
    // Show narration interface
    // Hide combat interface
  }
}
```

---

## Unified Game Flow

### Complete Player Journey

```
1. Player visits frontend
   ↓
2. Frontend → GET /campaigns
   ↓
3. Player selects "Dragons of Stormwreck Isle" + Fighter + "Thorin"
   ↓
4. Frontend → POST /game/start {campaign_id, character_class, character_name}
   ↓
5. Orchestrator:
   - Creates GameStateTree
   - Loads campaign template
   - Generates personalized opening with character
   - Returns session_id + opening narrative
   ↓
6. Frontend displays narrative in chat interface
   ↓
7. Player: "I explore the dock area"
   ↓
8. Frontend → POST /game/action {session_id, text: "I explore..."}
   ↓
9. Orchestrator:
   - Validates action with Rule Agent
   - Routes to Narrator Agent
   - Checks for combat trigger
   - Updates game tree
   ↓
10. If combat triggered:
    - Orchestrator switches to combat state
    - Frontend receives transition: "narration -> combat"
    - Frontend shows combat UI (using frontend-tmp combat interface)
    ↓
11. Combat actions go through same /game/action endpoint
    - Orchestrator routes to Combat Agent
    - Returns combat results
    ↓
12. When combat ends:
    - Orchestrator switches back to narration
    - Frontend receives transition: "combat -> narration"
    - Frontend shows narration UI
```

---

## Integrating Both Frontend UIs

### Strategy 1: Merge into Single React App

Create a unified component that switches between modes:

```typescript
// src/frontend/components/unified-game.tsx

export default function UnifiedGame() {
  const [gameState, setGameState] = useState<'narration' | 'combat'>('narration');
  const [sessionId, setSessionId] = useState<string>('');

  return (
    <div>
      {gameState === 'narration' && (
        <NarrationInterface
          sessionId={sessionId}
          onCombatStart={() => setGameState('combat')}
        />
      )}

      {gameState === 'combat' && (
        <CombatInterface
          sessionId={sessionId}
          onCombatEnd={() => setGameState('narration')}
        />
      )}
    </div>
  );
}
```

### Strategy 2: Use Tabs/Panels

Keep both interfaces visible with tabs:

```typescript
<Tabs value={activeTab}>
  <TabsList>
    <TabsTrigger value="story">Story</TabsTrigger>
    <TabsTrigger value="combat" disabled={!inCombat}>Combat</TabsTrigger>
  </TabsList>

  <TabsContent value="story">
    <NarrationInterface />
  </TabsContent>

  <TabsContent value="combat">
    <CombatInterface />
  </TabsContent>
</Tabs>
```

### Strategy 3: Iframe Embed (Quick & Dirty)

Embed frontend-tmp combat interface in an iframe:

```html
<!-- When combat starts -->
<iframe
  src="http://localhost:3001/combat"
  id="combat-iframe"
  style="display: none;"
></iframe>

<script>
  function switchToCombat(combatSessionId) {
    document.getElementById('narration-ui').style.display = 'none';
    document.getElementById('combat-iframe').style.display = 'block';

    // Send session ID to iframe
    iframe.contentWindow.postMessage({
      type: 'COMBAT_START',
      sessionId: combatSessionId
    }, '*');
  }
</script>
```

---

## Example: Complete Stormwreck Isle Start

### Step 1: Frontend makes request

```javascript
const response = await fetch('/api/game/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    campaign_id: 'stormwreck-isle',
    character_class: 'Fighter',
    character_name: 'Thorin Ironforge'
  })
});
```

### Step 2: Orchestrator returns

```json
{
  "session_id": "uuid-abc-123",
  "response": "You are Thorin Ironforge, a battle-hardened warrior whose sword has seen countless conflicts.\n\nYou stand on the deck of the merchant vessel *Compass Rose* as it approaches the mist-shrouded Stormwreck Isle...\n\nWhat do you do?",
  "campaign_info": {
    "campaign_id": "stormwreck-isle",
    "campaign_name": "Dragons of Stormwreck Isle",
    "starting_location": "Dragon's Rest Harbor",
    "initial_quest": "Explore Stormwreck Isle and discover why the dragons have called you here.",
    "difficulty": "beginner",
    "character_class": "Fighter",
    "character_name": "Thorin Ironforge"
  },
  "state": {
    "id": "node-1",
    "state_type": "narration",
    "metadata": {
      "is_root": true,
      "campaign_id": "stormwreck-isle",
      ...
    }
  }
}
```

### Step 3: Player takes action

```javascript
await fetch('/api/game/action', {
  method: 'POST',
  body: JSON.stringify({
    session_id: 'uuid-abc-123',
    text: 'I approach the dock and look for someone to talk to'
  })
});
```

### Step 4: Continue playing!

The orchestrator handles everything:
- ✅ Rule validation
- ✅ State transitions
- ✅ Agent routing
- ✅ Game tree updates

---

## Quick Start Testing

### 1. Start all services
```bash
docker-compose up --build
```

### 2. List campaigns
```bash
curl http://localhost/api/campaigns
```

### 3. Start Stormwreck Isle
```bash
curl -X POST http://localhost/api/game/start \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": "stormwreck-isle",
    "character_class": "Fighter",
    "character_name": "Thorin"
  }'
```

### 4. Take an action
```bash
curl -X POST http://localhost/api/game/action \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "text": "I explore the docks"
  }'
```

---

## Adding Custom Campaigns

Edit [campaign_loader.py](src/orchestrator/campaign_loader.py):

```python
CAMPAIGNS["my-custom-campaign"] = CampaignTemplate(
    id="my-custom-campaign",
    name="My Amazing Adventure",
    description="A custom campaign",
    opening_narrative="Your custom story here...",
    starting_location="Custom Location",
    initial_quest="Custom quest",
    difficulty="medium"
)
```

No database needed - campaigns are code-based templates!

---

## Summary

✅ **No pre-building needed** - Game tree is created automatically
✅ **Pre-loaded campaigns** - 5 ready-to-play adventures
✅ **Character customization** - Name + class integrated into story
✅ **API-driven** - Easy frontend integration
✅ **State management** - Automatic narration ↔ combat transitions
✅ **Both UIs work** - Can use frontend OR frontend-tmp

The system is ready to play Dragons of Stormwreck Isle right now! 🐉
