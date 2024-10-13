import "package:flutter/material.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:pyrin_app/core/page.dart";
import "package:pyrin_app/ui.dart";

class WordBadge extends StatelessWidget
{
    final String word;
    final int index;

    WordBadge({required this.word, required this.index});

    @override
    Widget build(BuildContext context)
    {
        final int n = index + 1;

        return Container(
          margin: const EdgeInsets.only(right: 10),
          padding: const EdgeInsets.only(top: 5, left: 20, right: 20, bottom: 1),
          decoration: BoxDecoration(
            color: PyrinColors.BLACK1_COLOR,
            borderRadius: BorderRadius.circular(50),
            border: Border.all(
              color: PyrinColors.TEXT_COLOR.withOpacity(0.04),
              width: 1,
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(word, style: Theme.of(context).textTheme.bodyMedium!.copyWith(color: PyrinColors.TEXT_COLOR, fontSize: 14)),
              Text("$n", style: Theme.of(context).textTheme.bodyMedium!.copyWith(color: PyrinColors.TEXT_COLOR.withOpacity(0.2), fontSize: 11)),
            ],
          ),
        );
    }
}

class CreateWalletPage extends StatefulWidget
{
    const CreateWalletPage({super.key});

    @override
    State<CreateWalletPage> createState() => _CreateWalletState();
}

class _CreateWalletState extends State<CreateWalletPage>
{
    @override
    Widget build(BuildContext context)
    {
        final List<String> words =
        [
          "insect", "announce", "ankle", "you", "crisp", "ordinary", "demise", "inflict", "feed", "write", "wet", "treat",
          "fit", "ten", "strategy", "ocean", "evolve", "wonder", "aisle", "monitor", "guitar", "burden", "mule", "sauce"
        ];

        return RoutePage(
          name: "Create Wallet",
          buttons: [
            PyrinElevatedButton(
              text: "Next",
              onClick: onNextClick,
              wide: true,
            )
          ],
          child: Column(
            children: [
              PyrinTitle(text: "Secret Recovery Phrase"),
              PyrinSubtitle(text: "Keep your Secret Recovery Phrase safe. It’s the key to accessing your wallet if you lose your device."),

              GridView.count(
                crossAxisCount: 3,
                childAspectRatio: 2.75,
                crossAxisSpacing: 0,
                mainAxisSpacing: 10,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                children: List.generate(24, (index) => WordBadge(word: words[index], index: index)),
              ),

              const Expanded(child: const SizedBox(height: 20)),
              PyrinFlatButton(
                text: "Copy to clipboard",
                rightIcon: "copy",
                onClick: (){},
              ),
            ],
          ),
        );
    }

    onNextClick()
    {

    }
}
