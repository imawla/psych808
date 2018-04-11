#!/usr/bin/python

html_text = [
	"<html>",
	"<head>",
	"	<title>Subject {:03d}: center slices</title>",
	"</head>",
	"<body>"
]

subj_id = 1

#print(html_text.format(subj_id))

for line in html_text:
	if line.find('{:03d}'):
		print(line.format(subj_id))
	else:
		print(line)

html_test = "\n".join(html_text)
print(html_text.format(subj_id))
