def expand(batch):
    batch_new = []
    for n, x in batch:
        for _ in range(n):
            batch_new.append(x)
    return batch_new

def compress(batch):
    prev = None
    batch_new = []
    for x in batch:
        if not batch_new or batch_new[-1][1] != x:
            batch_new.append((1, x))
        else:
            batch_new[-1] = (batch_new[-1][0] + 1, batch_new[-1][1])
    return batch_new

def escape(batch):
    # Format for python
    string = '['
    for n, x in batch:
        string += f'({n}, '
        text = ''
        piece = ''
        for c in x:
            if (c.isascii() and c.isalnum()) or c in [' ', ',', '.']:
                piece += c
            else:
                if text:
                    text += ' + '
                if piece:
                    text += f"'{piece}' + chr({ord(c)})"
                else:
                    text += f'chr({ord(c)})'
                piece = ''
        if piece:
            if text:
                text += ' + '
            text += f"'{piece}'"
            piece = ''
        if not text:
            text = "''"
        string += text
        string += '), '
    string += ']'

    # Escape for shell
    string = string.split('\'')
    string = '\'\\\'\''.join(string)
    string = '\'' + string + '\''

    return string
