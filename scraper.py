from bs4 import BeautifulSoup
import re
import requests
import json

def main():
	page_link = 'https://www.wired.com/story/ai-deepfakes-cant-save-us-duped/'

	# implement try catch

	page_response = requests.get(page_link, timeout=5)
	
	page_content = BeautifulSoup(page_response.content, "html.parser")
	
	paragraphs = page_content.find_all("p")
	clean_paragraphs = []

	images = page_content.find('figure').find_all('img', src=True)
	clean_images = []

	for pg in paragraphs:
		p = pg.text
		o = re.sub(r"\s+", " ", p)
		clean_paragraphs.append(o)
	print len(clean_paragraphs), "paragraphs found."

	for im in images:
		i = im['src']
		i.replace('&amp;', '&')
		clean_images.append(i)
	print len(clean_images), "images found."

	out = [clean_paragraphs, clean_images]

	with open('output.json', 'wb') as outfile:
		json.dump(out, outfile)

if __name__ == '__main__':
	main()