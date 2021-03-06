package com.fsuarez.ai.calc.needtoport;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class Readfile {
    
    private static final String resourcePath = new File("").getAbsolutePath().concat("/src/test/resources/");

    public Readfile(){}
    
    public static String[] fileLines(String fileName) throws FileNotFoundException, IOException {;
        BufferedReader lineReader = new BufferedReader(new FileReader(new File(resourcePath.concat(fileName))));
        BufferedReader lineNums = new BufferedReader(new FileReader(new File(resourcePath.concat(fileName))));
        String temp = "";
        int numLines = 0;
        while((temp =lineNums.readLine()) != null){
            numLines++;
        }
        String[] ret = new String[numLines];
        for(int i = 0; i < numLines; i++){
            ret[i] = lineReader.readLine();
        }
        return ret;
    }
    
    public static double[][] getFileArray(String fileName) throws IOException{
        String[] out = fileLines(fileName);
        String[] temp = out[0].split(",");
        double[][] ret = new double[out.length][temp.length];
        for(int r = 0; r < ret.length; r++){
            String[] outSplit = out[r].split(",");
            for(int c = 0; c < ret[0].length; c++){
                ret[r][c] = Double.parseDouble(outSplit[c]);
            }
        }
        return ret;
    }

}