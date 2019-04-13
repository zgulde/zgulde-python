def print_fn_docs(fn):
    print('``%s``' % fn.__name__)
    print('~' * (len(fn.__name__) + 4))
    print()
    print(strip_beginning_whitespace(fn.__doc__))
    print()

def strip_beginning_whitespace(paragraph):
    lines = [line.replace('    ', '', 1) for line in paragraph.split('\n')]
    return '\n'.join(lines)

if __name__ == '__main__':
    import zgulde.extend_pandas
    from zgulde.extend_pandas import data_frame_extensions, series_extensions

    print("Zgulde's pandas extensions")
    print("##########################")
    print()
    print(strip_beginning_whitespace(zgulde.extend_pandas.__doc__))
    print()

    print('Data Frame Extensions')
    print('=====================')
    print()

    for fn in data_frame_extensions:
        print_fn_docs(fn)

    print('Series Extensions')
    print('=====================')
    print()

    for fn in series_extensions:
        print_fn_docs(fn)
