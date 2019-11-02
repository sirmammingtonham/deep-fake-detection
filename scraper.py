from bs4 import BeautifulSoup
import re
import requests
import json

def main():
	page_link = 'https://medium.com/futuremag/fortnite-creative-mode-isn-t-for-everybody-9551c47d2de3'
	# this is the url that we've already determined is safe and legal to scrape from.

	page_response = requests.get(page_link, timeout=5)
	# here, we fetch the content from the url, using the requests library
	
	page_content = BeautifulSoup(page_response.content, "html.parser")
	#we use the html parser to parse the url content and store it in a variable.
	
	paragraphs = page_content.find_all("p")
	clean_paragraphs = []

	for pg in paragraphs:
		p = pg.text
		o = re.sub(r"\s+", " ", p)
		clean_paragraphs.append(o)
	print len(clean_paragraphs), "paragraphs found."

	with open('output.json', 'wb') as outfile:
		json.dump(clean_paragraphs, outfile)

if __name__ == '__main__':
	main()