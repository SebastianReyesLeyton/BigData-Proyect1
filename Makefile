start-ubuntu:
	clear
	python3 $(FILE) > results/$(RESULT).md

start-windows:
	cls
	py -m $(FILE) > results/$(RESULT).md