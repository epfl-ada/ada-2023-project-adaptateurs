black:
	black P2.ipynb
	black analysis/*.py
	black nlp/*.py
	black preprocessing/*.py

commit: 
	make black 
	git add P2.ipynb
	git add analysis/*.py
	git add nlp/*.py
	git add preprocessing/*.py
	git add .gitignore