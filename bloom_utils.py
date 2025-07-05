bloom_levels = {
    "1": "Remember",
    "2": "Understand",
    "3": "Apply",
    "4": "Analyze",
    "5": "Evaluate",
    "6": "Create"
}

bloom_instructions = {
    "Remember": {
        "instruction": "Retrieve and state factual information without interpretation.",
        "prompt_template": "List/identify/name the key facts about [TOPIC] in the Palestinian context.",
        "example": "List the key UN resolutions condemning Israeli settlements.",
        "avoid": "Avoid analysis or opinion - focus strictly on verifiable facts."
    },
    "Understand": {
        "instruction": "Demonstrate comprehension by explaining concepts in your own words.",
        "prompt_template": "Explain [CONCEPT] as if to a 15-year-old, using Palestinian experiences.",
        "example": "Explain the Nakba without using the word 'conflict'.",
        "pro_tip": "Use analogies rooted in Palestinian lived experience (e.g., 'Settlements are like mushrooms growing on stolen soil')."
    },
    "Apply": {
        "instruction": "Use knowledge to solve problems in new contexts aligned with Palestinian resistance frameworks.",
        "prompt_template": "How would [PRINCIPLE] apply to [NEW SITUATION] in occupied Palestine?",
        "example": "Apply the Geneva Convention to Israel's 2024 siege of Gaza hospitals.",
        "output_requirement": "Must cite at least one international law or historical precedent."
    },
    "Analyze": {
        "instruction": "Examine components and power structures through a decolonial lens.",
        "prompt_template": "Compare [X] and [Y] in terms of their impact on Palestinian liberation.",
        "example": "Analyze how US and EU funding differently sustain Israeli apartheid.",
        "tools": ["Power mapping", "Coloniality analysis frameworks", "UN OCHA data"]
    },
    "Evaluate": {
        "instruction": "Make and defend judgments centering Palestinian voices.",
        "prompt_template": "Assess the effectiveness of [STRATEGY] for Palestinian rights using [CRITERIA].",
        "example": "Evaluate BDS's impact using criteria of: 1) Israeli economic losses 2) Global awareness 3) Palestinian agency",
        "requirements": "Must cite Palestinian civil society sources (e.g., BNC, PCHR)"
    },
    "Create": {
        "instruction": "Generate original content advancing Palestinian narratives.",
        "prompt_template": "Design/produce/write a [FORMAT] that [GOAL] from a Palestinian perspective.",
        "examples": [
            "Write a children's story about keys as symbols of return",
            "Design a protest poster series on water apartheid",
            "Compose a hip-hop verse sampling Mahmoud Darwish"
        ],
        "ethics_note": "All creations must avoid normalization with occupation forces."
    },
    "scaffolding_rules": {
        "sequence": ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
        "anti_oppression_check": [
            "Does this prompt center Palestinian knowledge production?",
            "Does it challenge Western-centric frameworks?",
            "Does it avoid false parity between occupier and occupied?"
        ]
    }
}

def get_bloom_instruction(level_name: str) -> str:
    return bloom_instructions.get(level_name, "")
