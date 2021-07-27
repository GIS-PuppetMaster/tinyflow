def wrapper(fun):
    def decorator(self):
        fun(self)
        print('decorator')
    return decorator


class Father:
    def __init__(self):
        pass

    @wrapper
    def p(self):
        print('father')

class Child(Father):
    def __init__(self):
        super().__init__()

    def p(self):
        print('child')

c = Child()
c.p()