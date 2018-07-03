package com.lynn000000.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import scala.math.sqrt


// To run on EMR successfully
// aws s3 


object BusinessSimilarities1M {
  
  /** Load up a Map of Business IDs to Business names. */
  def loadBusinessNames() : Map[Int, String] = {
    
    // Handle character encoding issues:
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    // Create a Map of Ints to Strings, and populate it from u.item.
    var BusinessNames:Map[Int, String] = Map()
    
     val lines = Source.fromFile("Businesss.dat").getLines()
     for (line <- lines) {
       var fields = line.split("::")
       if (fields.length > 1) {
        BusinessNames += (fields(0).toInt -> fields(1))
       }
     }
    
     return BusinessNames
  }
  
  type BusinessRating = (Int, Double)
  type UserRatingPair = (Int, (BusinessRating, BusinessRating))
  def makePairs(userRatings:UserRatingPair) = {
    val BusinessRating1 = userRatings._2._1
    val BusinessRating2 = userRatings._2._2
    
    val Business1 = BusinessRating1._1
    val rating1 = BusinessRating1._2
    val Business2 = BusinessRating2._1
    val rating2 = BusinessRating2._2
    
    ((Business1, Business2), (rating1, rating2))
  }
  
  def filterDuplicates(userRatings:UserRatingPair):Boolean = {
    val BusinessRating1 = userRatings._2._1
    val BusinessRating2 = userRatings._2._2
    
    val Business1 = BusinessRating1._1
    val Business2 = BusinessRating2._1
    
    return Business1 < Business2
  }
  
  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]
  
  def computeCosineSimilarity(ratingPairs:RatingPairs): (Double, Int) = {
    var numPairs:Int = 0
    var sum_xx:Double = 0.0
    var sum_yy:Double = 0.0
    var sum_xy:Double = 0.0
    
    for (pair <- ratingPairs) {
      val ratingX = pair._1
      val ratingY = pair._2
      
      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingY
      numPairs += 1
    }
    
    val numerator:Double = sum_xy
    val denominator = sqrt(sum_xx) * sqrt(sum_yy)
    
    var score:Double = 0.0
    if (denominator != 0) {
      score = numerator / denominator
    }
    
    return (score, numPairs)
  }
  
  /** Our main function where the action happens */
  def main(args: Array[String]) {
    
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Create a SparkContext without much actual configuration
    // We want EMR's config defaults to be used.
    val conf = new SparkConf()
    conf.setAppName("BusinessSimilarities1M")
    val sc = new SparkContext(conf)
    
    println("\nLoading Business names...")
    val nameDict = loadBusinessNames()
    
    val data = sc.textFile("s3n://sundog-spark/ml-1m/ratings.dat")

    // Map ratings to key / value pairs: user ID => Business ID, rating
    val ratings = data.map(l => l.split("::")).map(l => (l(0).toInt, (l(1).toInt, l(2).toDouble)))
    
    // Emit every Business rated together by the same user.
    // Self-join to find every combination.
    val joinedRatings = ratings.join(ratings)   
    
    // At this point our RDD consists of userID => ((BusinessID, rating), (BusinessID, rating))

    // Filter out duplicate pairs
    val uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

    // Now key by (Business1, Business2) pairs.
    val BusinessPairs = uniqueJoinedRatings.map(makePairs).partitionBy(new HashPartitioner(100))

    // We now have (Business1, Business2) => (rating1, rating2)
    // Now collect all ratings for each Business pair and compute similarity
    val BusinessPairRatings = BusinessPairs.groupByKey()

    // We now have (Business1, Business2) = > (rating1, rating2), (rating1, rating2) ...
    // Can now compute similarities.
    val BusinessPairSimilarities = BusinessPairRatings.mapValues(computeCosineSimilarity).cache()
    
    //Save the results if desired
    //val sorted = BusinessPairSimilarities.sortByKey()
    //sorted.saveAsTextFile("Business-sims")
    
    // Extract similarities for the Business we care about that are "good".
    
    if (args.length > 0) {
      val scoreThreshold = 0.97
      val coOccurenceThreshold = 1000.0
      
      val BusinessID:Int = args(0).toInt
      
      // Filter for Businesss with this sim that are "good" as defined by
      // our quality thresholds above     
      
      val filteredResults = BusinessPairSimilarities.filter( x =>
        {
          val pair = x._1
          val sim = x._2
          (pair._1 == BusinessID || pair._2 == BusinessID) && sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold
        }
      )
        
      // Sort by quality score.
      val results = filteredResults.map( x => (x._2, x._1)).sortByKey(false).take(50)
      
      println("\nTop 50 similar Businesss for " + nameDict(BusinessID))
      for (result <- results) {
        val sim = result._1
        val pair = result._2
        // Display the similarity result that isn't the Business we're looking at
        var similarBusinessID = pair._1
        if (similarBusinessID == BusinessID) {
          similarBusinessID = pair._2
        }
        println(nameDict(similarBusinessID) + "\tscore: " + sim._1 + "\tstrength: " + sim._2)
      }
    }
  }
}