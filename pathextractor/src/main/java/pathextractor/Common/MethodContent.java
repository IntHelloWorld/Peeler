package pathextractor.Common;

import java.util.ArrayList;
import pathextractor.Features.ProgramRelation;

public class MethodContent {
    private ArrayList<ProgramRelation> relations;
    private String name;
    private long length;

    public MethodContent(ArrayList<ProgramRelation> relations, String name, long length) {
        this.relations = relations;
        this.name = name;
        this.length = length;
    }

    public ArrayList<ProgramRelation> getRelations() {
        return relations;
    }

    public String getName() {
        return name;
    }

    public long getLength() {
        return length;
    }

}
