import com.sun.tools.attach.VirtualMachine;

public class AgentRunner {

    /*
     * This class shows how to attach hotswap-agent.jar to a running JVM process and overload classes using "extraClasspath=" property via Hotswapper plugin.
     *
     * Lets assume that:
     *  args[0] contains pid of running JVM process or a runner class name we want to attach agent to
     *  "Usage: java -cp .:$JAVA_HOME/lib/tools.jar AgentRunner JVM_PID_OR_NAME"
     */
    public static void main(String[] args) {
        try {
            String pid = args[0];
            System.out.println("Main start");
            System.out.println(pid);
            System.out.println("Virtual Machine before attach");
            final VirtualMachine vm = VirtualMachine.attach(args[0]);
            System.out.println("Virtual Machine after attach");
            vm.loadAgentPath("/usr/local/scope/lib/libscope.so", null);
            System.out.println("Virtual Machine before detach");
            vm.detach();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
