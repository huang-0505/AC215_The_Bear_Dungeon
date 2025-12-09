"""
campaign_loader.py

Preloaded campaign templates for D&D adventures.
Provides initial game state setup for various campaigns.
"""

from typing import Dict, List, Optional
from game_state import GameStateTree, GameStateType, AgentType


class CampaignTemplate:
    """Represents a pre-defined D&D campaign"""

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        opening_narrative: str,
        starting_location: str,
        initial_quest: Optional[str] = None,
        difficulty: str = "medium",
    ):
        self.id = id
        self.name = name
        self.description = description
        self.opening_narrative = opening_narrative
        self.starting_location = starting_location
        self.initial_quest = initial_quest
        self.difficulty = difficulty

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "opening_narrative": self.opening_narrative,
            "starting_location": self.starting_location,
            "initial_quest": self.initial_quest,
            "difficulty": self.difficulty,
        }


# Pre-defined campaigns
CAMPAIGNS = {
    "stormwreck-isle": CampaignTemplate(
        id="stormwreck-isle",
        name="Dragons of Stormwreck Isle",
        description="A beginner-friendly adventure on a mysterious island inhabited by dragons. Perfect for new adventurers!",
        opening_narrative="""You stand on the deck of the merchant vessel *Compass Rose* as it approaches the mist-shrouded Stormwreck Isle.

The island looms before you, its rocky cliffs emerging from the fog like the spine of some great slumbering beast. Seabirds wheel overhead, their cries mixing with the crash of waves against stone.

Captain Tarak, a weathered half-orc with salt in her beard, points toward a sheltered cove. "Dragon's Rest," she says gruffly. "The only safe harbor on this cursed rock. They say good dragons watch over this place, but I've never seen 'em."

As the ship glides into the cove, you can make out wooden docks and a small cluster of buildings nestled against the cliff face. A tower rises from the settlement, its windows dark. Something about this place feels... expectant. As if the island itself has been waiting for you.

What do you do?""",
        starting_location="Dragon's Rest Harbor",
        initial_quest="Explore Stormwreck Isle and discover why the dragons have called you here.",
        difficulty="beginner",
    ),
    "classic-dungeon": CampaignTemplate(
        id="classic-dungeon",
        name="The Lost Mine of Phandelver",
        description="A classic dungeon crawl through ancient mines filled with treasure and danger.",
        opening_narrative="""The town of Phandalin lies before you, a small frontier settlement on the edge of civilization.

You've been hired by the dwarf Gundren Rockseeker to escort a wagon of supplies to the nearby town. The pay is good - 10 gold pieces each - and Gundren promised there would be more work once you arrived.

But Gundren and his companion, a human warrior named Sildar Hallwinter, rode ahead on horseback, saying they needed to "take care of business" in town. That was two days ago, and they haven't been seen since.

Now, as your wagon crests a hill, you see something troubling: two dead horses lying in the middle of the trail ahead. Black-feathered arrows protrude from their bodies. The horses wear Gundren's distinctive saddles.

The forest on either side of the trail is eerily quiet. What do you do?""",
        starting_location="The Triboar Trail",
        initial_quest="Find Gundren Rockseeker and discover what happened on the trail.",
        difficulty="medium",
    ),
    "wilderness-adventure": CampaignTemplate(
        id="wilderness-adventure",
        name="Tomb of Annihilation",
        description="Brave the deadly jungles of Chult to find and destroy the legendary death curse.",
        opening_narrative="""The humid air of Port Nyanzaru hits you like a wall as you disembark from the ship.

The bustling port city of Chult sprawls before you, a riot of color and sound. Merchants hawk exotic goods in a dozen languages, dinosaurs trumpet from the beast market, and the scent of strange spices fills the air.

But beneath the vibrant chaos, you sense something wrong. People walk with haunted eyes. Clerics huddle in worried groups. Death hangs over this place like a shroud.

The rumors are true: a death curse is spreading across Faerûn. Those who have been raised from the dead are wasting away, and resurrection magic no longer functions. The source of this curse lies somewhere in the deadly jungles of Chult - in the legendary Tomb of the Nine Gods.

A wizened guide approaches you, his reptilian eyes gleaming. "You seek the tomb, yes? Many have gone into the jungle. Few return. But Xandala knows the way... for a price."

What do you do?""",
        starting_location="Port Nyanzaru, Chult",
        initial_quest="Journey into the jungle and find the source of the death curse.",
        difficulty="hard",
    ),
    "gothic-horror": CampaignTemplate(
        id="gothic-horror",
        name="Curse of Strahd",
        description="Enter the mist-shrouded land of Barovia, where a vampire lord reigns supreme.",
        opening_narrative="""The fog came from nowhere, thick and cold, swallowing the road ahead.

One moment you were traveling through familiar countryside. The next, everything changed. The air grew heavy with dread. The trees twisted into gnarled, reaching forms. And when the mist finally parted, you found yourselves on a different road entirely - one flanked by towering, leafless trees and broken stones.

A weathered signpost emerges from the gloom:
*"Village of Barovia - 1 Mile"*

Behind you, where the road should lead back to the world you know, there is only more mist - thick, impenetrable, and somehow... hungry. There is no going back.

As you stand at this crossroads, you hear the distant howl of wolves, and somewhere in the darkness, the creak of a gate. A raven lands on the signpost, watching you with unsettling intelligence.

In your pack, you find a letter you don't remember receiving:
*"Hail to thee of might and valor. I am Kolyan Indirovich, Burgomaster of Barovia. My daughter Ireena has been afflicted by an evil so deadly that even the good people of our village cannot protect her. I beg you, come to our village and save my daughter from the devil Strahd..."*

The letter trails off in bloodstains.

What do you do?""",
        starting_location="The road to Barovia",
        initial_quest="Reach the village of Barovia and save Ireena from Count Strahd.",
        difficulty="hard",
    ),
    "planar-adventure": CampaignTemplate(
        id="planar-adventure",
        name="Planescape: Descent into Avernus",
        description="Save the city of Elturel from damnation in the Nine Hells.",
        opening_narrative="""The city of Baldur's Gate has always been a place of opportunity and danger, but today something is terribly wrong.

Refugees pour through the city gates, their eyes wide with terror. They speak of the impossible: the city of Elturel, Baldur's Gate's sister city to the east, has *vanished*.

Not destroyed. Not conquered. *Vanished*.

Where the gleaming city once stood, there is now only a massive crater filled with swirling darkness. And from that darkness come screams - thousands of screams.

The Flaming Fist mercenaries are overwhelmed. The city's priests desperately seek answers from their gods. And dark rumors spread: Elturel has fallen into Avernus, the first layer of the Nine Hells.

A High Observer of Torm approaches you, her face grim. "Elturel needs heroes. The city's leaders made a terrible pact, and now thousands of innocent souls are paying the price. Will you descend into the Hells themselves to save them?"

Behind her, through the window, you can see it: a tear in reality itself, pulsing with infernal light.

What do you do?""",
        starting_location="Baldur's Gate",
        initial_quest="Discover what happened to Elturel and find a way to save it from Hell.",
        difficulty="very hard",
    ),
}


class CampaignLoader:
    """Load and initialize campaigns for the game"""

    @staticmethod
    def list_campaigns() -> List[Dict]:
        """Get list of all available campaigns"""
        return [campaign.to_dict() for campaign in CAMPAIGNS.values()]

    @staticmethod
    def get_campaign(campaign_id: str) -> Optional[CampaignTemplate]:
        """Get a specific campaign by ID"""
        return CAMPAIGNS.get(campaign_id)

    @staticmethod
    def initialize_campaign(
        campaign_id: str, character_class: Optional[str] = None, character_name: Optional[str] = None
    ) -> Dict:
        """
        Initialize a campaign with optional character customization.

        Returns initial game state ready for the orchestrator.
        """
        campaign = CAMPAIGNS.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign '{campaign_id}' not found")

        # Customize opening narrative with character details
        opening = campaign.opening_narrative

        if character_class and character_name:
            character_intro = CampaignLoader._get_character_intro(character_class, character_name)
            opening = f"{character_intro}\n\n{opening}"
        elif character_class:
            opening = f"As a {character_class}, you find yourself on an adventure...\n\n{opening}"

        return {
            "campaign_id": campaign.id,
            "campaign_name": campaign.name,
            "initial_prompt": opening,
            "starting_location": campaign.starting_location,
            "initial_quest": campaign.initial_quest,
            "metadata": {
                "difficulty": campaign.difficulty,
                "character_class": character_class,
                "character_name": character_name,
            },
        }

    @staticmethod
    def _get_character_intro(character_class: str, character_name: str) -> str:
        """Generate character-specific introduction"""
        intros = {
            "Fighter": f"You are {character_name}, a battle-hardened warrior whose sword has seen countless conflicts.",
            "Wizard": f"You are {character_name}, a scholar of the arcane arts whose spellbook holds terrible power.",
            "Rogue": f"You are {character_name}, a cunning rogue whose quick wits are matched only by your quicker blade.",
            "Cleric": f"You are {character_name}, a devoted servant of the divine whose faith shields you from evil.",
            "Ranger": f"You are {character_name}, a master of the wilderness who moves like a shadow through the forest.",
            "Bard": f"You are {character_name}, a charismatic storyteller whose words can inspire armies or topple kingdoms.",
            "Paladin": f"You are {character_name}, a holy warrior whose oath binds you to justice and righteousness.",
            "Barbarian": f"You are {character_name}, a fierce warrior whose rage makes you a force of nature in battle.",
            "Monk": f"You are {character_name}, a martial artist who has honed your body into a living weapon.",
            "Druid": f"You are {character_name}, a guardian of nature who can call upon primal forces.",
            "Warlock": f"You are {character_name}, bound by a pact with an otherworldly patron who grants you eldritch power.",
            "Sorcerer": f"You are {character_name}, born with magic flowing through your very veins.",
        }

        return intros.get(character_class, f"You are {character_name}, an adventurer seeking glory and treasure.")
