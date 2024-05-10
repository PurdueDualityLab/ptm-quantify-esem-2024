# Import the data from the csv files
bottom <- read.csv("bottom_model_card_scores.csv")
top <- read.csv("top_model_card_scores.csv")


# Order by likes
bottom <- bottom[order(bottom$likes, decreasing = FALSE),]
top <- top[order(top$likes, decreasing = TRUE),]

# Box plot of first 1k rows
top_likes <- top$Documentation.Quality.Score[1:1000]
bottom_likes <- bottom$Documentation.Quality.Score[1:1000]
boxplot(
    top_likes,
    bottom_likes,
    names = c("Top 1k", "Bottom 1k"),
    # col = c("red", "blue"),
    main = "Documentation Quality Score for Likes",
    ylab = "Metric",
    xlab = "Group"
)

# Conduct a t-test on the top vs bottom
likes_results <- t.test(
    top_likes,
    bottom_likes,
    alternative = "greater",
    conf.level = 0.99
)

# Order by downloads
bottom <- bottom[order(bottom$downloads, decreasing = FALSE),]
top <- top[order(top$downloads, decreasing = TRUE),]

# Box plot of first 1k rows
top_downloads <- top$Documentation.Quality.Score[1:1000]
bottom_downloads <- bottom$Documentation.Quality.Score[1:1000]
boxplot(
    top_downloads,
    bottom_downloads,
    names = c("Top 1k", "Bottom 1k"),
    # col = c("red", "blue"),
    main = "Documentation Quality Score for Downloads",
    ylab = "Metric",
    xlab = "Group"
)

# Conduct a t-test on the top vs bottom
downloads_results <- t.test(
    top_downloads,
    bottom_downloads,
    alternative = "greater",
    conf.level = 0.99
)

# Order by num_downstream_repos
bottom <- bottom[order(bottom$num_downstream_repos, decreasing = FALSE),]
top <- top[order(top$num_downstream_repos, decreasing = TRUE),]

# Box plot of first 1k rows
top_num_downstream_repos <- top$Documentation.Quality.Score[1:1000]
bottom_num_downstream_repos <- bottom$Documentation.Quality.Score[1:1000]
boxplot(
    top_num_downstream_repos,
    bottom_num_downstream_repos,
    names = c("Top 1k", "Bottom 1k"),
    # col = c("red", "blue"),
    main = "Documentation Quality Score for Num Downstream Repos",
    ylab = "Metric",
    xlab = "Group"
)

# Conduct a t-test on the top vs bottom
downstream_results <- t.test(
    top_num_downstream_repos,
    bottom_num_downstream_repos,
    alternative = "greater",
    conf.level = 0.99
)

adjusted_p_values <- p.adjust(
    c(
        likes_results$p.value,
        downloads_results$p.value,
        downstream_results$p.value
    ),
    method = "holm"
)


# Calculate the effect size
library("lsr")
effect_likes <- cohensD(
    top_likes,
    bottom_likes
)

effect_downloads <- cohensD(
    top_downloads,
    bottom_downloads
)

effect_downstream <- cohensD(
    top_num_downstream_repos,
    bottom_num_downstream_repos
)

likes_results
downloads_results
downstream_results

adjusted_p_values

effect_likes
effect_downloads
effect_downstream
