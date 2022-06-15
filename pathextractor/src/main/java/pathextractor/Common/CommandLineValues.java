package pathextractor.Common;

import java.io.File;
import java.nio.file.Path;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

/**
 * This class handles the programs arguments.
 */
public class CommandLineValues {
    // For single file
    @Option(name = "--SHA", required = false)
    public String SHA = null;

    @Option(name = "--projectDir", required = false)
    public Path projectDir = null;

    @Option(name = "--classPath", required = false)
    public Path classPath = null;

    @Option(name = "--targetMethodNames", required = false)
    public String targetMethodNames = null;

    @Option(name = "--label", required = false)
    public String label = null;

    // For benchmark
    @Option(name = "--csvFile", required = false, forbids = "--targetMethodNames")
    public File csvFile = null;

    @Option(name = "--projectsDir", required = false, forbids = "--targetMethodNames")
    public Path projectsDir = null;

    @Option(name = "--num_threads", required = false)
    public int NumThreads = 32;

    // For both
    @Option(name = "--min_code_len", required = false)
    public int MinCodeLength = 1;

    @Option(name = "--max_code_len", required = false)
    public int MaxCodeLength = 10000;

    @Option(name = "--max_path_len", required = false)
    public int MaxPathLength = 50;

    @Option(name = "--max_path_count", required = false)
    public int MaxPathCount = 100;

    @Option(name = "--pretty_print", required = false)
    public boolean PrettyPrint = false;

    @Option(name = "--output_dir", required = false)
    public String outputDir = null;

    public CommandLineValues(String... args) throws CmdLineException {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            throw e;
        }
    }

    public CommandLineValues() {

    }
}