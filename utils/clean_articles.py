import re
import argparse

html_tags_pattern = re.compile(r'<[^>]*>')
spaces = re.compile(r'\s+')


def remove_html_tags(text):
    return html_tags_pattern.sub(text, '')


def remove_multiple_space(text):
    return spaces.sub(' ', text)


def remove_multiple_newlines(text):
    return text.replace('\r', '').replace('\n', '')


def replace_quotation(text):
    return text.replace('“', '"').replace('”', '"')


def main():
    with open(args.src, 'r') as in_file, \
         open(args.out, 'w') as out_file:
        articles = in_file.read().split('\n')
        # remove_article_place = re.compile(r'^[^–—-]*\s*[–—-]\s*')
        for article in articles:
            # article = remove_html_tags(article)
            article = remove_multiple_space(article)
            article = remove_multiple_newlines(article)
            article = replace_quotation(article)
            # article = remove_article_place.sub('', article)
            if article:
                print(article.strip('-').strip(), file=out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help="Filename to text file")
    parser.add_argument('out', type=str, help="Filename to output text file")
    args = parser.parse_args()
    main()
