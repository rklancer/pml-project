pml_write_files = function(x){
    if (!file.exists('submissions')) {
        dir.create('submissions')
    }
    n = length(x)
    for(i in 1:n){
        filename = paste0("submissions/problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

assignmentCases <- prep(read.csv('data/pml-testing.csv', na.strings=c("NA", "")))
answers <- predict(results$fit, newdata=assignmentCases)
pml_write_files(answers)
