def foo(*args, **kwargs):
    print('args = ', args)
    print('kwargs = ', kwargs)
    print(kwargs.get('a',None), '这是测试一下')
    print('---------------------------------------')

if __name__ == '__main__':
    foo(1, 2, 3, 4)
    foo(a=1, b=2, c=3)
    foo(1, 2, 3, 4, a=1, b=2, c=3)
    foo('a', 1, None, a=1, b='2', c=3)