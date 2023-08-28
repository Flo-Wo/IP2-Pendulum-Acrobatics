lqr:
	g++ -O3 -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		-I"/opt/homebrew/opt/boost@1.82/include" \
		-L"/opt/homebrew/opt/boost@1.82/lib" \
		-I"/opt/homebrew/opt/pinocchio/include" \
		-L"/opt/homebrew/opt/pinocchio/lib" \
		-I"./action_model/ActionModel" \
		-I"./datatypes/TorquesWithTime" \
		-I"./logger/Logger" \
		-I"./datatypes/State.hpp" \
		-I"./file_reader/FileReader" \
		-I"/usr/local/include/" \
		-L"/usr/local/lib" \
        ./action_model/ActionModel.out \
		./datatypes/TorquesWithTime.out \
		./logger/Logger.out \
		./file_reader/FileReader.out \
		-lboost_filesystem -lboost_system -lpinocchio -lcrocoddyl\
		./controller/lqr_tracker.cpp -o ./controller/lqr_tracker.out
mpc:
	g++ -O3 -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		-I"/opt/homebrew/opt/boost@1.82/include" \
		-L"/opt/homebrew/opt/boost@1.82/lib" \
		-I"/opt/homebrew/opt/pinocchio/include" \
		-L"/opt/homebrew/opt/pinocchio/lib" \
		-I"./action_model/ActionModel" \
		-I"./datatypes/TorquesWithTime" \
		-I"./logger/Logger" \
		-I"./file_reader/FileReader" \
		-I"/usr/local/include/" \
		-L"/usr/local/lib" \
        ./action_model/ActionModel.out \
		./datatypes/TorquesWithTime.out \
		./logger/Logger.out \
		./file_reader/FileReader.out \
		-lboost_filesystem -lboost_system -lpinocchio -lcrocoddyl\
		./controller/MPC.cpp -o ./controller/MPC.out
mpc_profiler:
	g++ -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		-I"/opt/homebrew/opt/boost@1.82/include" \
		-L"/opt/homebrew/opt/boost@1.82/lib" \
		-I"/opt/homebrew/opt/pinocchio/include" \
		-L"/opt/homebrew/opt/pinocchio/lib" \
		-I/opt/homebrew/Cellar/gperftools/2.10/include \
		-L/opt/homebrew/Cellar/gperftools/2.10/lib \
		-I"./action_model/ActionModel" \
		-I"./logger/Logger" \
		-I"./file_reader/FileReader" \
		-I"/usr/local/include/" \
		-L"/usr/local/lib" \
        ./action_model/ActionModel.out \
		./logger/Logger.out \
		./file_reader/FileReader.out \
		-lboost_filesystem -lboost_system -lpinocchio -lcrocoddyl -lprofiler\
		./controller/MPC.cpp -o ./controller/MPC_profiler.out
action_model:
	g++ -c -O3 \
		-I"/opt/homebrew/include/eigen3" \
		-I"/opt/homebrew/opt/boost@1.82/include" \
		-L"/opt/homebrew/opt/boost@1.82/lib" \
		-I"/opt/homebrew/opt/pinocchio/include" \
		-L"/opt/homebrew/opt/pinocchio/lib" \
		-I"/usr/local/include/" \
		-L"/usr/local/lib" \
		-I"./logger/Logger" \
        -lboost_filesystem -lboost_system -lpinocchio -lcrocoddyl \
        -std=c++14 \
		./logger/Logger.out \
        ./action_model/ActionModel.cpp \
		-o ./action_model/ActionModel.out

friction_model:
	g++ -O3 -c ./action_model/FrictionModel.hpp \
		-I"/opt/homebrew/include/eigen3" \
		--std=c++20 \

forward_integrator:
	g++ -O3 --std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		-I"/opt/homebrew/opt/pinocchio/include" \
		-L"/opt/homebrew/opt/pinocchio/lib" \
		-I"/opt/homebrew/opt/boost@1.82/include" \
		-L"/opt/homebrew/opt/boost@1.82/lib" \
		-I"./datatypes/State.hpp" \
		-I"./action_model/FrictionModel"\
		-I"./logger/Logger" \
		./datatypes/TorquesWithTime.out \
		-lpinocchio -lboost_filesystem -lboost_system \
		./logger/Logger.out \
		./state_observer/ForwardIntegrator.cpp \
		-o ./state_observer/ForwardIntegrator.out 

luenberg:
	g++ -O3 -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		-I"./datatypes/State.hpp" \
		-I"/opt/homebrew/opt/pinocchio/include" \
		-L"/opt/homebrew/opt/pinocchio/lib" \
		-I"/opt/homebrew/opt/boost@1.82/include" \
		-L"/opt/homebrew/opt/boost@1.82/lib" \
		-I"./friction_model/FrictionModel" \
		-I"./file_reader/FileReader" \
		-I"./logger/Logger" \
		-L"/usr/local/lib" \
		-lpinocchio -lboost_filesystem -lboost_system \
		./file_reader/FileReader.out \
		./logger/Logger.out \
		./state_observer/LuenbergObserver.cpp -o ./state_observer/LuenbergObserver.out
	
# -c

# Data types
state:
	g++ -c -O3 -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		./datatypes/state.hpp -o ./datatypes/State.out
torques_with_time:
	g++ -c -O3 -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		-I"./logger/Logger" \
		./logger/Logger.out \
		./datatypes/TorquesWithTime.cpp  -o ./datatypes/TorquesWithTime.out

# Helper classes
logger:
	g++ -c -O3 -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		./logger/Logger.cpp  -o ./logger/Logger.out
file_reader:
	g++ -c -O3 -std=c++14 \
		-I"/opt/homebrew/include/eigen3" \
		./file_reader/FileReader.cpp  -o ./file_reader/FileReader.out

# bindings
bindings:
	g++ -O3 -Wall -shared -std=c++14 -fPIC \
	-g \
	-I"/opt/homebrew/include/pybind11" \
	-I"/opt/homebrew/include/eigen3" \
	-I"/opt/homebrew/opt/boost@1.82/include" \
	-L"/opt/homebrew/opt/boost@1.82/lib" \
	-I"/opt/homebrew/opt/pinocchio/include" \
	-L"/opt/homebrew/opt/pinocchio/lib" \
	-I"/Users/florianwolf/miniconda3/envs/croc20/include/" \
	-I"/Users/florianwolf/miniconda3/envs/croc20/include/python3.9" \
	-lpinocchio -lboost_filesystem -lboost_system \
	./python_bindings/bindings.cpp \
	./state_observer/LuenbergObserver.cpp \
	-o pendulum_acrobatics_cpp.so \
  	-L/Users/florianwolf/miniconda3/envs/croc20/lib -lpython3.9
# $(python3 -m pybind11 --includes) \
# -lpython3.9
# -I"./state_observer/LuenbergObserver.hpp" \
# $(/Users/florianwolf/miniconda3/envs/croc20/bin/python -m pybind11 --includes) \
