import "package:flutter/material.dart";

class SectionTitle extends StatelessWidget
{
    static const double HEIGHT = 30;

    final String name;

    SectionTitle({Key? key, required this.name});

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: double.infinity,
          height: HEIGHT,
          child: Text(name,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge!.copyWith(fontWeight: FontWeight.w600, fontSize: 20)
          ),
        );
    }
}


class SectionContainer extends StatelessWidget
{
    static const double PADDING = 20;

    final String name;
    final Widget child;

    SectionContainer({Key? key, required this.name, required this.child});

    @override
    Widget build(BuildContext context)
    {
      return Stack(
        children: [
          SectionTitle(name: name),
          Container(
              margin: const EdgeInsets.only(top: PADDING * 0.5 + SectionTitle.HEIGHT, bottom: PADDING),
              child: child
          ),
        ],
      );
    }
}

