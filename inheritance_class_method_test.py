class Base:
    def __init__(cls, x):
        print('Initializing base class')
        cls.x = x
        cls.__init__Directional(x)
    def plot(self):
        print('Plotting from base:', self.x)
    def hell_yeah(self):
        print("hell yeah")
    # @classmethod
    # def directional(cls, x):
    #     obj1 = cls(x)
    #     sectors = [obj1]
    class Directional():
        def __init__(self, sectors):
            print('Initializing direction class')
            self.sectors = sectors

        def plot(self):
            for sector_model in self.sectors:
                print('plotting from directional base')
                sector_model.plot()

        def synthesize(self):
                return 0
        # return Directional(sectors)
class Derived(Base if directional==False else Base.Directions):
    def __init__(self, x, directional=True):
        print('Initialising Derived class')
        self.x = x

        # if not check:
        #     print("entering if")
        #     Base.__init__(self, x)
        # else:
        #     print("entering else")
        #     Base.Directional.__init__(self, x)
        print(dir(self))

    def predict(self):
        print('Predicted:', self.x - 1)

obj = Derived(5)
obj.plot()
obj.hell_yeah()


