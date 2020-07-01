import initializeTest
import forwardTest
import costTest
import backwardTest
import updateTest

print("Initializing parameters: ")
initializeTest.initialize_test()
print()

print("Propagating forward: ")
forwardTest.forward_test()
print()

print("Computing cost: ")
costTest.cost_test()
print()

print("Propagating backward: ")
backwardTest.backward_test()
print()

print("Updating parameters: ")
updateTest.update_test()
print()
