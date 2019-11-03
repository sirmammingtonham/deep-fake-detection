from bs4 import BeautifulSoup
import cv2
import numpy as np
import urllib
import re
import requests
import json

def main():
	# page_link = 'https://www.wired.com/story/ai-deepfakes-cant-save-us-duped/'
	page_link = 'https://www.engadget.com/2019/10/11/deepfake-celebrity-impresonations/'

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
	print (len(clean_paragraphs), "paragraphs found.")

	for im in images:
		url = im['src']
		url.replace('&amp;', '&')
		print(url)

		urllib.request.urlretrieve(url, "temp.jpg")

		# Get user supplied values
		imagePath = "temp.jpg"
		cascPath = "haarcascade_frontalface_default.xml"

		# Create the haar cascade
		faceCascade = cv2.CascadeClassifier(cascPath)

		# Read the image
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Detect faces in the image
		faces = faceCascade.detectMultiScale(
		    gray,
		    scaleFactor=1.1,
		    minNeighbors=5,
		    minSize=(30, 30)
		)

		print("Found {0} faces!".format(len(faces)))

		# Draw a rectangle around the faces
		# for (x, y, w, h) in faces:
		#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# cv2.imshow("Faces found", image)
		# cv2.waitKey(0)

		if len(faces) > 0:
			clean_images.append(url)

	print (len(clean_images), "images found.")

	out = [clean_paragraphs, clean_images]

	with open('output.json', 'w') as outfile:
		json.dump(out, outfile)

if __name__ == '__main__':
	main()