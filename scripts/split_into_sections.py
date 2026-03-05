import pandas as pd
import regex as re
import spacy
nlp = spacy.load("en_core_web_trf")
import en_core_web_trf

splits = {'workforce': 'interview_transcripts/workforce_transcripts.csv', 'creatives': 'interview_transcripts/creatives_transcripts.csv', 'scientists': 'interview_transcripts/scientists_transcripts.csv'}
df = pd.read_csv("hf://datasets/Anthropic/AnthropicInterviewer/" + splits["creatives"])


# Define topics with their questions
TOPICS = {
    'intro': [],
    'job_description': [
        "ould you tell me a bit about your creative work and what a typical project looks like",
        "tell me a bit about your creative work?",
        "what does a typical project look like for you?",
    ],
    'ai_integration': [
        "Where and when do you find yourself reaching for AI tools?",
        "how AI fits into your creative process",
        "alk me through how AI fits into your creative process",
    ],
    'specific_project': [
        "a specific example of a recent situation",
        "how did that feel for you",
        "you describe a specific recent project where you used AI?",
        "specific recent project where you used AI",
        "What does that process look like when you do use AI",
        "an you describe a specific recent example",
        "Can you describe what that experience was like for you?",
        "How did that feel different from using AI",
    ],
    'decision_making': [
        "how would you describe that dynamic",
        "ow would you describe that collaboration",
        "driving the creative direction",
        "ho's driving the creative decisions",
        "ho or what is driving the creative decisions",
        "driving the creative decisions?",
    ],
    'changed_aspects': [
        "hat aspects of your creative work have changed the most",
    ],
    'concerns': [
        "hat concerns, if any, do you have",
    ],
    'future': [
        "Looking ahead, how do you see AI",
        "how do you see AI's role in your creative work evolving",
    ],
    'closing': [
        "Those are all the questions I had prepared.",
        "Before we wrap up,",
        "Something we haven't covered yet?",
    ]
}


def identify_topic(text):
    """Identify which topic category a text snippet belongs to."""
    for topic, questions in TOPICS.items():
        if any(quest.lower() in text for quest in questions):
            return topic
    return None

def split_themes(all_text):
    """Split conversation into a keyed dictionary by topic."""
    result = {topic: {'user': [], 'assistant': []} for topic in TOPICS.keys()}
    result['unclassified'] = {'user': [], 'assistant': []}

    current_speaker = None
    current_topic = 'intro'
    current_text = ''

    for segment in re.splititer('(Assistant: |\nUser: |\nAI: )', all_text.lstrip(), flags=re.DOTALL):
        segment = segment.strip()
        if not segment:
            continue
        # Check if this is a speaker marker
        if segment in ['Assistant:', 'AI:']:
            # Save previous user text if any
            if current_text and current_speaker == 'user':
                result[current_topic]['user'].append(current_text.strip())
            current_speaker = 'assistant'
            current_text = ''
        elif segment == 'User:':
            # Save previous assistant text if any
            if current_text and current_speaker == 'assistant':
                result[current_topic]['assistant'].append(current_text.strip())
            current_speaker = 'user'
            current_text = ''
        else:
            # This is actual content
            current_text += '\n|\n' + segment if current_text else segment

            # If this is a user message, check if it matches a new topic
            if current_speaker == 'user':
                detected_topic = identify_topic(segment)
                if detected_topic and detected_topic != current_topic:
                    # Save current accumulated text to old topic before switching
                    if current_text.strip():
                        result[current_topic]['user'].append(current_text.strip())
                        current_text = ''
                    # Switch to new topic
                    current_topic = detected_topic

    # Save any remaining text
    if current_text.strip():
        if current_speaker == 'user':
            result[current_topic]['user'].append(current_text.strip())
        elif current_speaker == 'assistant':
            result[current_topic]['assistant'].append(current_text.strip())

    # Remove empty topics
    #result = {k: v for k, v in result.items() if v['user'] or v['assistant']}

    return result


def extract_all_topics(df, text_column='text'):
    """Extract all topics and add each as a separate column."""

    # Initialize columns for each topic
    for topic in TOPICS.keys():
        df[f'{topic}_user'] = ''
        df[f'{topic}_assistant'] = ''

    for idx, row in df.iterrows():
        conversation_dict = split_themes(row[text_column])
        for topic, content in conversation_dict.items():
            if topic in TOPICS:  # Skip 'unclassified'
                df.at[idx, f'{topic}_user'] = '\n\n'.join(content['user'])
                #df.at[idx, f'{topic}_assistant'] = '\n\n'.join(content['assistant'])

    return df


# Usage:
df = extract_all_topics(df, text_column='text')

