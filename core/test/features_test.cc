// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the features code.

#include "gtest/gtest.h"

#include <algorithm>

#include "../features.h"

// Checks the basic word shape function.
TEST(WordFeatures, BasicWordShape) {
    EXPECT_EQ("<shape5>=all-digit", features::basic_word_shape("248"));
    EXPECT_EQ("<shape5>=all-upper", features::basic_word_shape("ABC"));
    EXPECT_EQ("<shape5>=all-lower", features::basic_word_shape("abc"));
    EXPECT_EQ("<shape5>=capitalized", features::basic_word_shape("Abc"));
    EXPECT_EQ("<shape5>=other", features::basic_word_shape("a1"));
}

// Checks the word identity function.
TEST(WordFeatures, WordIdentity) {
    EXPECT_EQ("<id>=aardvark", features::word_identity("aardvark"));
}

// Checks the prefix function.
TEST(WordFeatures, Prefix) {
    EXPECT_EQ("<prefix0>=", features::prefix("singing", 0));
    EXPECT_EQ("<prefix1>=s", features::prefix("singing", 1));
    EXPECT_EQ("<prefix4>=sing", features::prefix("singing", 4));
    EXPECT_DEATH(features::prefix("singing", 8), "");
}

// Checks the suffix function.
TEST(WordFeatures, Suffix) {
    EXPECT_EQ("<suffix0>=", features::suffix("singing", 0));
    EXPECT_EQ("<suffix1>=g", features::suffix("singing", 1));
    EXPECT_EQ("<suffix3>=ing", features::suffix("singing", 3));
    EXPECT_DEATH(features::suffix("singing", 8), "");
}

// Checks the contains-digit function.
TEST(WordFeatures, ContainsDigit) {
    EXPECT_EQ("<contains-digit>=t", features::contains_digit("aardvark0"));
    EXPECT_EQ("<contains-digit>=f", features::contains_digit("aardvark"));
}

// Checks the contains-hyphen function.
TEST(WordFeatures, ContainsHyphen) {
    EXPECT_EQ("<contains-hyphen>=t", features::contains_hyphen("aard-vark"));
    EXPECT_EQ("<contains-hyphen>=f", features::contains_hyphen("aardvark"));
}

// Checks the is-capitalized function.
TEST(WordFeatures, IsCapitalized) {
    EXPECT_EQ("<is-capitalized>=t", features::is_capitalized("Aardvark"));
    EXPECT_EQ("<is-capitalized>=f", features::is_capitalized("aardvark"));
    EXPECT_EQ("<is-capitalized>=f", features::is_capitalized("aArdvark"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
