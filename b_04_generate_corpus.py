import os
import re
import wikipedia

def sanitize_filename(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '_', input_string)

# The Wikipedia search API returns at most 500 results per request.
def generate_corpus(search_term="human rights", num_articles=500, output_dir="all_articles"):

    # Generate a corpus of Wikipedia articles and save them as text files.

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved = 0
    search_results = wikipedia.search(search_term, results=num_articles)
    
    for i, title in enumerate(search_results, 1):
        try:
            # Get the page content
            page = wikipedia.page(title, auto_suggest=False)
            
            # Save to file
            sanitized = sanitize_filename(title)
            filename = f"{sanitized}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page.content)
            saved += 1
                
        except Exception as e:
            print(f"Error processing '{title}': {str(e)}")
            continue
    
    print(f"\nCompleted! Saved {saved} articles!")

if __name__ == "__main__":
    generate_corpus()
