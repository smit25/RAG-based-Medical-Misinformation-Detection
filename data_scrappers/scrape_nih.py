import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin

def get_page_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_text(text):
    if not text:
        return ""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_home_text(soup):
    home_text = []
    
    # Find the main content area - try different selectors to be robust
    main_content = soup.select_one('.main-content, main, #main-content, article')
    if not main_content:
        main_content = soup  # Fallback to entire document
    
    # Extract paragraphs from the main content
    paragraphs = main_content.select('p:not(.footer p):not(nav p)')
    
    # Filter out empty paragraphs and those in navigation/footer
    for para in paragraphs:
        text = para.text.strip()
        if text and len(text) > 20:  # Avoid very short texts
            # Skip navigation and footer paragraphs
            is_nav_or_footer = False
            for parent in para.parents:
                if parent.name == 'nav' or (parent.get('class') and any(c in ['footer', 'nav', 'menu'] for c in parent.get('class', []))):
                    is_nav_or_footer = True
                    break
            
            if not is_nav_or_footer:
                home_text.append({"name": clean_text(text)})
    
    return home_text

def extract_links(soup, base_url):
    links = []
    visited_urls = set()
    visited_names = set()
    
    # Find diabetes-related links
    diabetes_links = []
    
    # Find all links in content areas
    for a_tag in soup.select('a'):
        href = a_tag.get('href', '')
        if not href or href.startswith('#') or href.startswith('javascript:'):
            continue
            
        # Normalize URL
        full_url = urljoin(base_url, href)
        
        # Skip external links, focus on NIH domain
        if not full_url.startswith('https://www.niddk.nih.gov/'):
            continue
            
        # Identify diabetes-related content
        if '/diabetes' in full_url.lower() or '/health-information/diabetes' in full_url.lower():
            link_text = clean_text(a_tag.text)
            if link_text and len(link_text) > 3 and full_url not in visited_urls and link_text not in visited_names:
                visited_urls.add(full_url)
                visited_names.add(link_text)
                diabetes_links.append({
                    "name": link_text,
                    "url": full_url
                })
    
    return diabetes_links

def extract_link_data(url):
    try:
        content = get_page_content(url)
        if not content:
            return ""
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the content container - try multiple selectors to be robust
        content_container = soup.select_one('.main-content, article, main, #content, .content')
        if not content_container:
            content_container = soup  # Fallback
        
        # Remove navigation, footer, etc.
        for element in content_container.select('nav, footer, .footer, .navigation, .menu, script, style, meta'):
            element.decompose()
        
        # Extract all meaningful text
        text_parts = []
        for element in content_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = clean_text(element.text)
            if text and len(text) > 10:  # Skip very short elements
                text_parts.append(text)
        
        return '\n'.join(text_parts)
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return ""

def main():
    base_url = "https://www.niddk.nih.gov/health-information/diabetes"
    content = get_page_content(base_url)
    
    if not content:
        print("Failed to fetch the home page.")
        return
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract home text and links
    home_text = extract_home_text(soup)
    home_links = extract_links(soup, base_url)
    
    # Extract data from each link
    for i, link in enumerate(home_links):
        print(f"Processing link {i+1}/{len(home_links)}: {link['name']}")
        link_data = extract_link_data(link['url'])
        home_links[i]['link_data'] = link_data
        
        # Be nice to the server
        time.sleep(2)
    
    result = {
        "home_text": home_text,
        "home_links": home_links
    }
    
    # Save to JSON file
    with open('nih.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    
    print(f"Data saved to nih.json")

if __name__ == "__main__":
    main()