package pathextractor.Common;

public class ProgressBar {

    private int items;
    private int index = 1;
    private String finish;
    private String unFinish;

    private final int PROGRESS_SIZE = 50;

    public ProgressBar(int items) {
        this.items = items;
    }

    private String getNChar(int num, char ch) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < num; i++) {
            builder.append(ch);
        }
        return builder.toString();
    }

    public void printProgress() throws InterruptedException {
        int progress = PROGRESS_SIZE * index / items;

        finish = getNChar(progress, '█');
        unFinish = getNChar(PROGRESS_SIZE - progress, '─');
        String target = "Progress:" + String.format("%.1f%%[%s%s]", (double) index / items * 100, finish, unFinish);
        System.out.print(target);
        if (index != items)
            System.out.print(getNChar(target.length(), '\b'));
        if (index == items)
            System.out.print('\n');

        Thread.sleep(1000);
        index++;
    }
}
