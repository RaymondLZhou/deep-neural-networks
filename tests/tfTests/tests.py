import initializeTest
import forwardTest
import costTest

print("Initializing parameters: ")
initializeTest.initialize_test()
print()

print("Propagating forward: ")
forwardTest.forward_test()
print()

print("Computing cost: ")
costTest.cost_test()
print()
