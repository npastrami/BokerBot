import eval7

if __name__ == "__main__":

    test_deck = eval7.Deck()

    for i in range(47):
        test_deck.deal(1)

    print("Can we take the length of the deck", len(test_deck))
    print("Can we print the deck??:", [card for card in test_deck])

    test_deck.shuffle()

    print("Can we take the length of the deck", len(test_deck))
    print("Can we print the deck??:", [card for card in test_deck])