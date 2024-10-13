import "dart:math" as math;
import "package:flutter/material.dart";
import "package:flutter/widgets.dart";
import "package:flutter_svg/flutter_svg.dart";

import "../core/token.dart";
import "../ui.dart";

List<Gradient> _gradients =
[
  LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFB01EFF), Color(0xFFE1467C)],
  ),
  LinearGradient(
    begin: Alignment.topRight,
    end: Alignment.bottomLeft,
    colors: [Color(0xFFD079EE), Color(0xFF8A88FB)],
  ),
  LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFC99FFF), Color(0xFF981ED2)],
  ),
  LinearGradient(
    begin: Alignment.bottomCenter,
    end: Alignment.topCenter,
    colors: [Color(0xFF9B23EA), Color(0xFF5F72BD)],
  ),
  LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFB39FFF), Color(0xFF6A1ED2)],
  ),
  LinearGradient(
    begin: Alignment.topRight,
    end: Alignment.bottomLeft,
    colors: [Color(0xFF4300B1), Color(0xFFA531DC)],
  ),
  LinearGradient(
    begin: Alignment.bottomLeft,
    end: Alignment.topRight,
    colors: [Color(0xFF764BA2), Color(0xFF667EEA)],
  ),
];

Gradient getRandomGradient()
{
  return _gradients[math.Random().nextInt(_gradients.length)];
}

class TokenIcon extends StatelessWidget {
  final double size;
  final String? symbol;

  const TokenIcon({super.key, this.size = 22, this.symbol});

  @override
  Widget build(BuildContext context) {
    if (symbol == "PYI") {
      return SvgPicture.asset(
        "assets/pyrin-coin.svg",
        width: size,
        height: size,
      );
    }

    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        gradient: getRandomGradient(),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Stack(
        alignment: Alignment.center,
        children: [
          SvgPicture.asset(
            "assets/icons/diamond.svg",
            width: 12,
            height: 12,
            colorFilter: ColorFilter.mode(
              PyrinColors.LIGHT_TEXT_COLOR.withOpacity(0.5),
              BlendMode.srcIn,
            ),
          ),
        ],
      ),
    );
  }
}

class TokensList extends StatelessWidget
{
  final void Function(Token token)? onClick;

  TokensList({this.onClick});

  @override
  Widget build(BuildContext context)
  {
    final double width = MediaQuery.sizeOf(context).width;

    double generateRandomBalance()
    {
      final math.Random random = math.Random();

      // Define the range
      final double min = 0.01;  // Minimum balance
      final double max = 1000.0; // Maximum balance

      // Generate a random number within the range
      return min + (random.nextDouble() * (max - min));
    }

    final List<Token> tokens =
    [
      Token("Pyrin", "PYI", 150000000000),
      Token("PyrinCoin", "PYC", (generateRandomBalance() * 1e8).round()),
      Token("PyrinX", "PYX", (generateRandomBalance() * 1e8).round()),
    ];

    return PyrinListView<Token>(
        items: tokens,
        onItemClick: (token)
        {
          onClick?.call(token);
        },
        itemBuilder: (context, token)
        {
          return PyrinListViewItemBuilder(
            title: token.symbol,
            subtitle: token.name,
            leading: Text((token.amount / 1e8).toStringAsFixed(2)),
            icon: PyrinCircleIcon(
              darker: true,
              child: TokenIcon(symbol: token.symbol),
            ),
          );
        }
    );
  }
}
