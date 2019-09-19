
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.io.*;
import com.csvreader.CsvReader;
import java.awt.BorderLayout;
import com.csvreader.CsvWriter;
import weka.classifiers.Evaluation;
import weka.core.*;
import weka.classifiers.functions.Logistic;
import weka.core.converters.ArffLoader;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;
import weka.classifiers.evaluation.ThresholdCurve;
import java.lang.Exception;

public class csv
{
    //the percentage in csv file should be transferd into decimal in advance
    static String[] Header;
    static int headerLen;
    static String csvFilePath = "E:/model/model/data2.csv";
    static String trainFilePath="E:/model/train.arff";
    static String testFilePath="E:/model/test.arff";
    static int[] numericList={2,10,11,12};
    static int[] charList={3,7,9};
    static int[] dummyList={4,5,6,8,13};

    /*get headers of raw data

    in:None
    return: the headers of csv file(String[])

     */
    public String[] getHeaders()
    {
        try
        {
            CsvReader data = new CsvReader(csvFilePath);
            int columns = data.getHeaderCount();
            String[] headers=new String[columns];
            data.readHeaders();
            headers=data.getHeaders();

            return headers;
        }
        catch (IOException e)
        {
            System.out.println(e);

        }
        return null;
    }


    /*read raw data csv file

    in:None
    return: the main body of csv file(ArrayList<String>)

     */
    public static ArrayList<String[]> readcsv()
    {
        try
        {
            ArrayList<String[]> csvList = new ArrayList<String[]>();
            CsvReader data = new CsvReader(csvFilePath);

            data.readHeaders(); //

            while (data.readRecord()) {
                csvList.add(data.getValues());
            }
            return csvList;
        }
        catch (IOException e) {
            System.out.println(e);
            return null;
        }

    }


    /*get the single numeric feature as an array

    in:the main body of csv file(ArrayList<String>); index of numeric feature(int)
    return: numeric feature array(double[])

     */
    public static double[] getNumFeature(ArrayList<String[]> csvList,int numericIndex)
    {
        double avg;
        double sum = 0;
        ArrayList<Double> thisfeature = new ArrayList<Double>();
        for (int row = 0; row < csvList.size(); row++) {

            String cell = csvList.get(row)[numericIndex];
            //System.out.println(cell);

            if ((!cell.equals("n.a.")) && !cell.equals("n.s.")) {
                double cellnum = Double.valueOf(cell);
                sum = sum + cellnum;
                thisfeature.add(cellnum);
            }

        }
        //System.out.println(thisfeature);
        avg = sum / thisfeature.size();
        //System.out.println(avg);

        thisfeature.clear();
        for (int row = 0; row < csvList.size(); row++) {

            String cell = csvList.get(row)[numericIndex];
            //System.out.println(cell);

            if ((cell.equals("n.a.")) || cell.equals("n.s.")) {
                cell = String.valueOf(avg);
                double cellnum = Double.valueOf(cell);
                thisfeature.add(cellnum);
            } else {
                double cellnum = Double.valueOf(cell);
                thisfeature.add(cellnum);

            }

        }
        //System.out.println(thisfeature);
        ////////transfer values to double[]/////////////
        double[] feature=new double[thisfeature.size()];
        for (int a=0;a<thisfeature.size();a++)
        {
            double num=(double)thisfeature.get(a);
            feature[a]=num;
        }
        return  feature;
    }

    /*get the single character feature as an array

    in:the main body of csv file(ArrayList<String>); index of character feature(int)
    return: character feature array(String[])

    */
    public static String[] getCharFeature(ArrayList<String[]> csvList,int charIndex)
    {
        ArrayList<String> thisfeature = new ArrayList<String>();
        for (int row = 0; row < csvList.size(); row++) {

            String cell = csvList.get(row)[charIndex];
            //System.out.println(cell);
            thisfeature.add(cell);
        }

        String[] feature=new String[thisfeature.size()];
        for (int a=0;a<thisfeature.size();a++)
        {
            String character=thisfeature.get(a);
            feature[a]=character;
        }
        return  feature;
    }


    /*get the descriptive statistics of a single numeric feature

    in: numeric feature array(double[])
    return: descriptive statistics of a single numeric feature (double[])

    */
    public static double[] describe(double[] numFeatureArray)
    {
        /////////////////////computing average/////////////////
        double[] describe = new double[6];
        int len = numFeatureArray.length;
        double sum = 0;
        for (int i = 0; i < len; i++) {
            sum += numFeatureArray[i];
        }
        double dAve = sum / len;
        describe[0] = dAve;

        ///////////////computing standard deviation///////////
        double dVar = 0;
        for (int i = 0; i < len; i++) {
            dVar += ( numFeatureArray[i] - dAve) * ( numFeatureArray[i] - dAve);
        }
        double std = Math.sqrt(dVar / len);
        describe[1] = std;

        ///////////////finding max and min///////////
        double max = numFeatureArray [0];
        for (int i = 1; i < len; i++) {
            if ( numFeatureArray[i] > max) {
                max =  numFeatureArray[i];
            }
        }

        double min =  numFeatureArray[0];
        for (int i = 1; i < len; i++) {
            if ( numFeatureArray[i] < min) {
                min =  numFeatureArray[i];
            }
        }
        describe[2] = max;
        describe[3] = min;

        //////////////////computing Kurtosis and Skewness////////////////
        double sum_4 = 0;
        double sum_3 = 0;
        for (int i = 1; i < len; i++) {
            sum_3 += Math.pow(( numFeatureArray[i] - dAve), 3);
            sum_4 += Math.pow(( numFeatureArray[i] - dAve), 4);
        }
        double kurtosis = ((sum_4 / (len - 1)) / Math.pow(std, 4)) - 3;
        double skewness = ((sum_3 / (len - 1)) / Math.pow(std, 3));
        describe[4] = skewness;
        describe[5] = kurtosis;

        return describe;

    }

    /*get the frequency of a single character feature

    in: character feature array(String[])
    return: frequency of every type of this feature (ArrayList<ArrayList<String>>)

    */
    public static ArrayList<ArrayList<String>> frequency(String[] charFeatureArray)
    {
        ArrayList<String> feature=new ArrayList<String>();
        feature.add(charFeatureArray[0]);
        ArrayList<Integer> featureClassNum=new ArrayList<Integer>();
        featureClassNum.add(1);
        ArrayList<String> featureClassChar=new ArrayList<String>();
        for (int x=0;x<charFeatureArray.length;x++)
        {
            String cell=charFeatureArray[x];
            int containNum=0;
            for(int i=0;i<feature.size();i++)
            {
                boolean contain=cell.equals(feature.get(i));
                if (contain==true){containNum=containNum+1;}
            }


            if(containNum==0)//need to add new type
            {
                feature.add(cell);
                featureClassNum.add(1);
            }
            else //type exist , count plus one
            {
                for(int k=0;k<feature.size();k++)
                {
                    if(cell.equals(feature.get(k)))
                    {
                        int now=featureClassNum.get(k)+1;
                        featureClassNum.set(k,now);
                    }
                }
            }

        }

        for (int x:featureClassNum)
        {
            String number=String.valueOf(x);
            featureClassChar.add(number);
        }

        ArrayList<ArrayList<String>> frequency=new ArrayList<>();
        frequency.add(feature);
        frequency.add(featureClassChar);
        return frequency;

    }


    /*data process for extreme values in a single numeric feature

    in: numeric feature array(double[]),descriptive statistic of this numeric feature(double[])
    return: None

    */
    public static void numericProcess(double[]  numFeatureArray,double[] describe)
    {
        int len= numFeatureArray.length;
        for (int i=0;i<len;i++)
        {
            if ( numFeatureArray[i]>describe[0]+3*describe[1])
            {
                numFeatureArray[i]=describe[0]+3*describe[1];
            }

            if ( numFeatureArray[i]<describe[0]-3*describe[1])
            {
                numFeatureArray[i]=describe[0]-3*describe[1];
            }
        }
    }


    /*get no-duplicated random index numbers

    in: sample index array(double[])
    return: sample index of training set and test set(ArrayList<int[]>)

    */
    public static ArrayList<int[]> indexGroup(double[] allIndex)
    {
        int len = allIndex.length;

        int randomNum = len;//pick up half of the data as training set
        int randomNumTest = (int) (0.05 * len);//pick up test set
        int[] trainingIndex = new int[randomNum];
        int[] testIndex = new int[randomNumTest];

        Random rand = new Random();

        trainingIndex[0] = rand.nextInt(len);
        for (int i = 1; i < randomNum; i++) {
            trainingIndex[i] = rand.nextInt(len);
            for (int j = 0; j < i; j++) {
                while (trainingIndex[i] == trainingIndex[j]) {
                    i--;
                }
            }
        }
        //System.out.println(Arrays.toString(trainingIndex));
        /*int testIndexID=0;
        for (int i =0;i<allIndex.length;i++)
        {   int total=0;
            for (int x:trainingIndex){
                total= i==x?total+1:total;
            }
            if (total==0)
            {
                testIndex[testIndexID]=i;
                testIndexID=testIndexID+1;
            }
        }
        */

        Random randTest = new Random();

        testIndex[0] = randTest.nextInt(len);
        for (int i = 1; i < randomNumTest; i++) {
            testIndex[i] = rand.nextInt(len);
            for (int j = 0; j < i; j++) {
                while (testIndex[i] == testIndex[j]) {
                    i--;
                }
            }
        }

        ArrayList<int[]> indexGroup=new ArrayList<>();
        indexGroup.add(trainingIndex);
        indexGroup.add(testIndex);
        return indexGroup;

    }


    /* use the random number above to get a data set

    in: an array of index (int[])
    return: a data set (ArrayList<String[]>)

    */
    public static ArrayList<String[]> divideSet(int[] setIndex)
    {
        try
        {
            ArrayList<String[]> csvList = new ArrayList<>();
            CsvReader data = new CsvReader(csvFilePath);

            data.readHeaders(); //

            while (data.readRecord())
            {
                csvList.add(data.getValues());
            }//end while
            ArrayList<String[]> setrow = new ArrayList<>();
            for (int row :setIndex)
            {
                String[] cell = csvList.get(row);
                //System.out.println(Arrays.toString(cell));
                setrow.add(cell);
            }
            return setrow;
        }
        //end try
        catch (IOException e)
        {
            System.out.println(e);
            return null;
        }

    }

    public static void main(String args[])throws Exception
    {
        csv datafile=new csv();
        Header=datafile.getHeaders();
        headerLen=Header.length;
        ArrayList<String[]> allset=datafile.readcsv();
        FileOutputStream fileOutput = new FileOutputStream(new File("E:/model/describe.txt"));
        PrintStream txtprint = new PrintStream(fileOutput);
        //////////////////print numeric feature descriptive statistics//////////////
        for (int i:numericList)
        {
            double[] values=datafile.getNumFeature(allset,i);
            double[] featureDescribe=datafile.describe(values);
            datafile.numericProcess(values,featureDescribe);
            //System.out.println(Arrays.toString(values));
            txtprint.println(Header[i]+" [avg,std,max,min,skewness,kurtosis]:"+Arrays.toString(featureDescribe)+"\r\n");
        }
        //////////////////print characteristic feature frequency summary//////////////
        for (int i:charList)
        {
            String[] values=datafile.getCharFeature(allset,i);
            ArrayList<ArrayList<String>> frequencyList=datafile.frequency(values);
            txtprint.println("=========Feature "+Header[i]+" frequency summary=======");
            txtprint.println(frequencyList.get(0)+":"+"\r\n"+frequencyList.get(1));
        }

        /////////////////divide into training set and test set after data process/////////
        double[] values=datafile.getNumFeature(allset,0);
        ArrayList<int[]> TrainTestIndex=datafile.indexGroup(values);
        ArrayList<String[]> trainingSet=datafile.divideSet(TrainTestIndex.get(0));


        /////////////////logistic regression/////////////////
        modelSet set=new modelSet();
        modelSet test=new modelSet();
        ArrayList<ArrayList<double[]>> newSet= modelSet.newSet(trainingSet,TrainTestIndex.get(1));

        set.writeArff(newSet.get(0),trainFilePath);
        set.writeArff(newSet.get(1),testFilePath);
        double[][] coef=set.regression(trainFilePath,testFilePath);

        /////////////////get model score//////////////////////
        score testScore=new score();
        String[] rank=testScore.score(coef,newSet.get(1));
        testScore.writeResult(rank);

    }

}


class modelSet extends csv
{
    static double IVthreshold=0;
    /* get the useful features index according to IV valuesuse

    in: training set(ArrayList<String[]>),test set sample index array(int[])
    return: a data set after WOE replacement and feature selection(ArrayList<ArrayList<double[]>>)

    */
    public static ArrayList<ArrayList<double[]>> newSet(ArrayList<String[]> trainingSet,int[] testSetIndex)
    {
        modelSet train= new modelSet();
        double[] trainFlag = train.getNumFeature(trainingSet, 13);

        int trgood = 0;
        int trbad = 0;
        for (double x : trainFlag) {
            if (x == 1) {
                //System.out.println("trainflag=1");
                trbad = trbad + 1;
            } else {
                //System.out.println("trainflag=0");
                trgood = trgood + 1;
            }
        }

        double[] IVList = new double[numericList.length];

//////////WOE calculation of numeric data////////////////////////////
        ArrayList<double[]> everyFeature=new ArrayList<>();
        ArrayList<double[]> everyNumFeature=new ArrayList<>();
        for (int i:numericList) {
            double[] trainFeature = train.getNumFeature(trainingSet, i);
            double[] featureDes = modelSet.describe(trainFeature);
            modelSet.numericProcess(trainFeature, featureDes);
            float trgood1 = 0;
            float trgood2 = 0;
            float trgood3 = 0;
            float trgood4 = 0;
            float trbad1 = 0;
            float trbad2 = 0;
            float trbad3 = 0;
            float trbad4 = 0;
            double avg = featureDes[0];
            double sigma = featureDes[1];
            int[] groupID = new int[trainFeature.length];

            ////////////////////divide into 4 groups////////////////////////
            for (int index = 0; index < trainFeature.length; index++) {
                if (trainFeature[index] <= avg - 0.75 * sigma) {
                    groupID[index] = 1;
                    if (trainFlag[index] == 1) {
                        trbad1 = trbad1 + 1;
                    } else {
                        trgood1 = trgood1 + 1;
                    }
                }

                if (trainFeature[index] > avg - 0.75 * sigma && trainFeature[index] <= avg) {
                    groupID[index] = 2;
                    if (trainFlag[index] == 1) {
                        trbad2 = trbad2 + 1;
                    } else {
                        trgood2 = trgood2 + 1;
                    }
                }

                if (trainFeature[index] > avg && trainFeature[index] <= avg + 0.75 * sigma) {
                    groupID[index] = 3;
                    if (trainFlag[index] == 1) {
                        trbad3 = trbad3 + 1;
                    } else {
                        trgood3 = trgood3 + 1;
                    }
                }

                if (trainFeature[index] > avg + 0.75 * sigma) {
                    groupID[index] = 4;
                    if (trainFlag[index] == 1) {
                        trbad4 = trbad4 + 1;
                    } else {
                        trgood4 = trgood4 + 1;
                    }
                }

            }

            if (trgood1 < 1 || trbad1 < 1 || trgood2 < 1 || trbad2 < 1) {
                trgood2 = trgood2 + trgood1;
                trbad2 = trbad2 + trbad1;
            }
            if (trgood4 < 1 || trbad4 < 1 || trgood3 < 1 || trbad3 < 1) {
                trgood3 = trgood3 + trgood4;
                trbad3 = trbad3 + trbad4;
            }

            float[] groupGood = new float[]{trgood1, trgood2, trgood3, trgood4};
            float[] groupBad = new float[]{trbad1, trbad2, trbad3, trbad4};
            double[] WOE = new double[6];//WOE[4]and WOE[5] is specially defined for WOE replacing

            /*
            System.out.println(group1+","+trgood1+","+trbad1);
            System.out.println(group2+","+trgood2+","+trbad2);
            System.out.println(group3+","+trgood3+","+trbad3);
            System.out.println(group4+","+trgood4+","+trbad4);
            System.out.println(Arrays.toString(groupID));
            */

            ////////////////////calculate WOE////////////////////////
            for (int groupnum = 0; groupnum < 4; groupnum++) {
                //System.out.println(groupGood[groupnum]);
                //System.out.println(groupBad[groupnum]);
                if (groupGood[groupnum] > 0 && groupBad[groupnum] > 0) {
                    double trgoodper = groupGood[groupnum] / trgood;
                    double trbadper = groupBad[groupnum] / trbad;
                    //System.out.println(trgoodper);
                    //System.out.println(trbadper);
                    WOE[groupnum] = Math.log((groupGood[groupnum] / trgood) / (groupBad[groupnum] / trbad));
                } else {
                    WOE[groupnum] = 0;
                    if (groupnum == 0 || groupnum == 1) {
                        WOE[4] = Math.log((groupGood[1] / trgood) / (groupBad[1] / trbad));
                    } else {
                        WOE[5] = Math.log((groupGood[2] / trgood) / (groupBad[2] / trbad));
                    }
                }
            }
            //System.out.println(Arrays.toString(WOE));
            //System.out.println(Arrays.toString(groupID));
            ////////////////replace feature values with WOE valuse/////////////////
            for (int index = 0; index < trainFeature.length; index++)
            {
                int ID = groupID[index]-1;
                if (WOE[ID] == 0)
                {
                    trainFeature[index] = WOE[4] == 0 ? WOE[5] : WOE[4];
                    //System.out.println("WOE is 0 and replace with "+trainFeature[index]);
                }
                else{trainFeature[index]=WOE[ID];}
            }


            ////////////////////calculate IV////////////////////////
            double IV = 0;
            for (int index = 0; index < WOE.length - 2; index++) {
                IV = IV + (WOE[index] * ((groupGood[index] / trgood) - (groupBad[index] / trbad)));
            }
            if (IV>IVthreshold)
            {
                everyFeature.add(trainFeature);
                everyNumFeature.add(trainFeature);
            }

        }

///////////////////WOE calculation for characteristic data/////////////////////

        for (int i: charList)
        {
            String[] trainFeature = train.getCharFeature(trainingSet, i);
            ArrayList<ArrayList<String>> frequencyList=train.frequency(trainFeature);

            int numWOE=frequencyList.get(0).size();
            int[] groupGood=new int[numWOE];
            int[] groupBad=new int[numWOE];
            int[] classNum=new int[trainFeature.length];
            double[] newchar=new double[trainFeature.length];
            double[] charWOE=new double[numWOE];
            //go through this char feature content
            for (int index=0;index<trainFeature.length;index++ )
            {
                String cell= trainFeature[index];
                String cellFlag=Double.toString(trainFlag[index]);

                //System.out.println("cellfalg"+cellFlag);
                //go through frequency List to summary good and bad
                for (int x=0;x<numWOE;x++)
                {
                    String check=frequencyList.get(0).get(x);
                    if (cell.equals(check))
                    {
                        classNum[index]=x;
                        switch (cellFlag)
                        {
                            case "0.0":groupGood[x]=groupGood[x]+1;break;
                            case "1.0":groupBad[x]=groupBad[x]+1;break;
                        }
                    }

                }
            }

            for (int x=0;x<numWOE;x++)
            {
                double perGood=(double)groupGood[x] / (double)trgood;
                double perBad=(double)groupBad[x] / (double)trbad;
                //System.out.println(perGood);
                //System.out.println(perBad);
                charWOE[x]= Math.log(perGood/perBad);
            }

            //System.out.println(Arrays.toString(charWOE));
            //System.out.println(Arrays.toString(classNum));

            for (int x=0;x<trainFeature.length;x++)
            {
                int thisClassNum=classNum[x];
                newchar[x]=charWOE[thisClassNum];
            }
            //System.out.println("WOE"+Arrays.toString(newchar));
            double charIV= 0;
            for (int index = 0; index < numWOE; index++) {
                double perGood=(double)groupGood[index] / (double)trgood;
                double perBad=(double)groupBad[index] / (double)trbad;
                charIV = charIV + (charWOE[index] * (perGood-perBad));
            }

            if (charIV>IVthreshold)
            {
                everyFeature.add(newchar);
            }
        }

        ////////////add dummy variable feature to all features////////////
        /*for (int i: dummyList)
        {
            double[] dummyFeature=train.getNumFeature(trainingSet,i);
            everyFeature.add(dummyFeature);
        }
*/
        everyFeature.add(trainFlag);
        //System.out.println("everyFeature size:"+everyFeature.size());


        //////////////////sort IV and pick up feature//////////////////////
       /* int[] allIndex = new int[IVList.length];
        int startPoint = (int) Math.ceil(0.2 * (IVList.length)) - 1;
        int uselen = headerLen - startPoint - 3;
        System.out.println("we use feature :"+uselen);
        int[] useIndex = new int[uselen+1];

        /////////////////only use 80% of the features as efficient features//////////

        HashMap map = new HashMap();
        for (int i = 0; i < IVList.length; i++) {
            map.put(IVList[i], i);
        }// put the index into map

        Arrays.sort(IVList);
        for (int i = 0; i < IVList.length; i++) {
            allIndex[i] = (int) map.get(IVList[i]);
        }

        System.arraycopy(allIndex, startPoint, useIndex, 0, uselen);
        //System.out.println(Arrays.toString(IVList));
        //System.out.println(Arrays.toString(allIndex));
        useIndex[uselen]=headerLen - 3;//add the flag column
        //System.out.println(Arrays.toString(useIndex));
*/

        ///////////////////use all features //////////////////////
        int[] useIndex=new int[everyFeature.size()];
        for (int i=0;i<useIndex.length;i++)
        {
            useIndex[i]=i;
        }
        //System.out.println("useIndex:"+Arrays.toString(useIndex));

        /////////////////////get new trainset and testset/////////////////
        ArrayList<ArrayList<double[]>> featureCol=new ArrayList<>();
        featureCol.add(everyFeature);
        ArrayList<ArrayList<double[]>> setLines=new ArrayList<>();

        ArrayList<double[]> Lines=new ArrayList<>();
        for (int row=0;row<trainFlag.length;row++)
        {
            double[] line=new double[useIndex.length];

            for (int col=0;col<useIndex.length;col++)
            {
                int lineNum=useIndex[col];
                //System.out.println(Arrays.toString(featureCol.get(listNum).get(lineNum)));
                //System.out.println(lineNum);
                double value=featureCol.get(0).get(lineNum)[row];
                //System.out.println(value);
                line[col]=value;
            }//get a single line
            //System.out.println(Arrays.toString(line));
            Lines.add(line);//get all lines of a single set

        }
        System.out.println("train set size:"+Lines.size());
        setLines.add(Lines);

        ArrayList<double[]> testLines = new ArrayList<>();
        for (int i =0;i<testSetIndex.length;i++)
        {
            int thisIndex=testSetIndex[i];
            double[] thisLines=Lines.get(thisIndex);
            testLines.add(thisLines);
        }
        setLines.add(testLines);
        System.out.println("test set size:"+testLines.size());

        /////////////////////////////print setLines///////////////////////
/*
        for (int y=0;y<setLines.get(0).size();y++)
        {
            double[] aline=setLines.get(0).get(y);
            //System.out.println("train"+Arrays.toString(aline));
            System.out.println("train"+aline[0]);
        }

        for (int y=0;y<setLines.get(1).size();y++)
        {
            double[] aline=setLines.get(1).get(y);
           //System.out.println("test"+Arrays.toString(aline));
            System.out.println("test"+aline[0]);
        }
*/
        return setLines;
    }


    /* write arff file(the feature statements should be modified according to features' atrribute )

    in: data set(ArrayList<double[]>),arff file path(String)
    return:None
    out: arff file of data set

    */
    public void writeArff(ArrayList<double[]> setData,String filePath)throws IOException
    {
        FileWriter writer = new FileWriter(filePath);
        BufferedWriter bw = new BufferedWriter(writer);
        bw.write("@RELATION"+" "+"Logistic_Regression"+"\r\n");
        for(int i = 1;i<setData.get(0).length;i++)
        {
            bw.write("@ATTRIBUTE"+" "+"attribute"+i+" "+"NUMERIC"+"\r\n");
        }
/*
        for (int k=0;k<4;k++)
        {
            bw.write("@ATTRIBUTE"+" "+"dummy"+k+" "+" {1.0,0.0}"+"\r\n");
        }
 */
        ///////////write flag//////////////////////
        bw.write("@ATTRIBUTE " + "flag" + " {1.0,0.0}"+"\r\n"); //flag as discrete variable

        ///////////write main data//////////////////////
        bw.write("@DATA"+"\r\n");

        for (int i=0;i<setData.size();i++){
            double[] instance=setData.get(i);
            String strIns=Arrays.toString(instance);
            strIns=strIns.substring(1,strIns.length()-1);
            //System.out.println(strIns);
            bw.write(strIns+"\r\n");
        }
        bw.close();
        writer.close();
    }


    /* logistic regrssion

   in: training set arff file path(String),test set arff file path(String)
   return:coeficients of results(double[][])
   out: roc graph; confusion matrix

   */
    static double[][] regression(String trainPath,String testPath) throws Exception {

        File trainFile = new File(trainPath); //train set file path
        ArffLoader trainLoader = new ArffLoader();
        trainLoader.setFile(trainFile);
        Instances insTrain =trainLoader.getDataSet(); // read train file into weka
        insTrain.setClassIndex(insTrain.numAttributes() - 1);
        Logistic logic=new Logistic();

        logic.buildClassifier(insTrain);//build classifier
        Evaluation eval = new Evaluation(insTrain); //build a model evaluation

        File testFile = new File(testPath);//test data file
        ArffLoader testLoader = new ArffLoader();

        testLoader.setFile(testFile );
        Instances insTest =testLoader.getDataSet(); // read test data into weka
        insTest.setClassIndex(insTest.numAttributes()-1);
        double sum = insTest.numInstances();
/*
        for(int i=0;i<sum;i++){

            Instance ins = insTest.instance(i);
            double prediction=logic.classifyInstance(ins);

            if (prediction==ins.classValue())
            {
                System.out.println("No.\t" + i + "\t" + ins.classValue() + " RIGHT");
            }
            else
            {
                System.out.println("No.\t" + i + "\t" + ins.classValue() + " WRONG");
            }

        } */

        FileOutputStream fileOutput = new FileOutputStream(new File("E:/model/regression_evaluation.txt"));
        PrintStream txtprint = new PrintStream(fileOutput);
        double[][] coef=logic.coefficients();
        eval.evaluateModel(logic, insTest);//evaluate classifier with test data
        txtprint.println("Logistic Regression");
        txtprint.println("Area under ROC curve: "+eval.areaUnderROC( 1));
        double Gini=(eval.areaUnderROC( 1)-0.5)*2;
        txtprint.println("Gini is :"+Gini);

        ThresholdCurve tc = new ThresholdCurve();

        Instances result = tc.getCurve(eval.predictions(), 0);


        //through Instances to obtain TP、FP array

        //int tpIndex = result.attribute(ThresholdCurve.TP_RATE_NAME).index();
        //int fpIndex = result.attribute(ThresholdCurve.FP_RATE_NAME).index();
        //double [] tpRate = result.attributeToDoubleArray(tpIndex);
        //double [] fpRate = result.attributeToDoubleArray(fpIndex);


        String resultString=logic.toString();
        System.out.println(resultString);
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString("(Area under ROC = " +
                Utils.doubleToString(ThresholdCurve.getROCArea(result), 4) + ")");
        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();
        vmc.addPlot(tempd);

        String plotName = vmc.getName();
        final javax.swing.JFrame jf =
                new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vmc, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter()
        {
            public void windowClosing(java.awt.event.WindowEvent e)
            {
                jf.dispose();
            }
        });
        jf.setVisible(true);

        return coef;
    }

}

class score extends modelSet {

    static int groupNumber = 10;
    static double cutoff = 0.30;

    /* logistic regrssion

    in: coeficients of results(double[][]); test set(ArrayList<double[]>)
    return:score of test set smaples(String[])

    */
    public String[] score(double[][] coef, ArrayList<double[]> testSet) throws FileNotFoundException{
        double[] coefList = new double[coef.length];
        for (int i = 0; i < coef.length; i++) {
            double thiscoef = coef[i][0];
            coefList[i] = thiscoef;
        }

        int[] testFlag = new int[testSet.size()];
        int len = testSet.get(0).length;
        for (int index = 0; index < testSet.size(); index++) {
            testFlag[index] = (int) testSet.get(index)[len - 1];
        }

        double[] badPro = new double[testSet.size()];
        double[] scoreArray = new double[testSet.size()];
        int[] predictClass = new int[testSet.size()];
        int[] testFlagArray = new int[testSet.size()];
        int flagIndex = testSet.get(0).length - 1;
        int tegood = 0;
        int tebad = 0;
        for (int i = 0; i < testSet.size(); i++) {
            double sum = 0;
            //////////////summury good and bad people in test set///////////////
            testFlagArray[i] = (int) testSet.get(i)[flagIndex];
            if (testFlagArray[i] == 1) {
                tegood = tegood + 1;
            } else {
                tebad = tebad + 1;
            }

            //////////////get score of every people in test set///////////////

            for (int m = 1; m < coefList.length; m++) {
                sum = sum + coefList[m] * testSet.get(i)[m - 1];
            }
            scoreArray[i] = sum + coefList[0];
        }

        for (int i = 0; i < scoreArray.length; i++) {
            badPro[i] = Math.pow(Math.E, scoreArray[i]) / (Math.pow(Math.E, scoreArray[i]) + 1);
            if (badPro[i] > cutoff) {
                predictClass[i] = 1;
            } else {
                predictClass[i] = 0;
            }
        }
        //predictClass is the final class of sample

        ///////////////confusion matrix///////////////
        double[][] confusion = new double[2][2];
        int act1pre1 = 0;
        int act0pre1 = 0;
        int act1pre0 = 0;
        int act0pre0 = 0;
        for (int i = 0; i < scoreArray.length; i++) {
            switch (predictClass[i]) {
                //prediction is 1
                case 1:
                    if (testFlag[i] == 1) {
                        //System.out.println("prediction of falg 1 is correct");
                        act1pre1 = act1pre1 + 1;
                    } else {
                        //System.out.println("prediction of falg 1 is wrong");
                        act0pre1 = act0pre1 + 1;
                    }
                    break;

                //prediction is 0
                case 0:
                    if (testFlag[i] == 1) {
                        //System.out.println("prediction of falg 0 is wrong");
                        act1pre0 = act1pre0 + 1;
                    } else {
                        //System.out.println("prediction of falg 0 is correct");
                        act0pre0 = act0pre0 + 1;
                    }
                    break;
            }
        }
        FileOutputStream fileOutput = new FileOutputStream(new File("E:/model/confusion_matrix.txt"));
        PrintStream txtprint = new PrintStream(fileOutput);
        txtprint.println("=========confusion matrix=========");
        txtprint.println("     0       1------predicted flag");
        txtprint.println("0    "+act0pre0+"    "+act0pre1);
        txtprint.println("1    "+act1pre0+"    "+act1pre1);
        confusion[0][0] = act0pre0;
        confusion[0][1] = act0pre1;
        confusion[1][0] = act1pre0;
        confusion[1][1] = act1pre1;

        HashMap map = new HashMap();
        for (int i = 0; i < scoreArray.length; i++) {
            map.put(scoreArray[i], i);
        }// put the index into map

        String[] rank = new String[badPro.length];
        for (int i = 0; i < badPro.length; i++) {
            if (badPro[i] <= 0.1) {
                rank[i] = "A";
            }
            if (badPro[i] <= 0.2 && badPro[i] > 0.1) {
                rank[i] = "B";
            }
            if (badPro[i] < 0.3 && badPro[i] > 0.2) {
                rank[i] = "C";
            }
            if (badPro[i] >= 0.3) {
                rank[i] = "D";
            }
        }

        /*
        double gap = (scoreArray[scoreArray.length - 1] - scoreArray[0]) / groupNumber;
        double[] accGood = new double[groupNumber];
        double[] accBad = new double[groupNumber];
        double[] groupClass = new double[scoreArray.length];

        /////////////get the group division of test sample///////////
        for (int i = 0; i < scoreArray.length; i++) {
            int groupBelong = (int) Math.ceil((scoreArray[i] - scoreArray[0]) / gap);
            groupBelong = groupBelong == 0 ? 1 : groupBelong;
            groupClass[i] = groupBelong;

            int sampleID = scoreIndex[i];
            double thisFlag = testFlagArray[sampleID];
            if (thisFlag == 1.0) {
                accGood[groupBelong - 1] = accGood[groupBelong - 1] + 1;
            } else {
                accBad[groupBelong - 1] = accBad[groupBelong - 1] + 1;
            }
        }

        /////////////get KS value list////////////
        double[] KSvalues = new double[accGood.length];
        for (int i = 0; i < accGood.length; i++) {
            accGood[i] = accGood[i] / tegood;
            accBad[i] = accBad[i] / tebad;
            double KS = accGood[i] - accBad[i];
            KSvalues[i] = KS;
        }

        //System.out.println("KS distance array:");
        //System.out.println(Arrays.toString(KSvalues));
        Arrays.sort(KSvalues);
        double maxKS = KSvalues[KSvalues.length - 1];
        System.out.println("and the max KS is :" + maxKS);
        */
        return rank;
    }


    /* write the score into csv file

     in: score of test set smaples(String[])
     return:None
     out: csv file of test samples' credit score

     */
    public static void writeResult(String[] rank) {
        String resultPath = "E:/model/result.csv";
        try {
            CsvWriter csvWriter = new CsvWriter(resultPath, ',', Charset.forName("utf-8"));

            String[] headers = {"sample index", "rank"};
            csvWriter.writeRecord(headers);
            for (int i = 0; i < rank.length; i++) {
                String[] content = {String.valueOf(i), rank[i]};
                csvWriter.writeRecord(content);
            }
            csvWriter.close();
        }
        catch (IOException e) {
            System.out.println(e);
        }

    }
}
