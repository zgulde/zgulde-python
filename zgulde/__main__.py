import json

info = {
  'name': 'Zach Gulde',
  'email': {
    'personal': 'zach.gulde@gmail.com',
    'work': 'zach@codeup.com'
  },
  'twitter': '@zgulde',
  'github': 'https://github.com/zgulde',
  'url': 'https://zgul.de',
}

print(json.dumps(info, indent=4))
