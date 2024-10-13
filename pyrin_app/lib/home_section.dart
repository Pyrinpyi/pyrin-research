import "dart:math" as math;
import "dart:ui" as ui;
import "package:intl/intl.dart";
import "package:flutter/material.dart";
import "package:flutter/services.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:provider/provider.dart";
import "package:pyrin_app/core/addressbook.dart";
import "package:pyrin_app/core/wallet_provider.dart";
import "package:pyrin_app/send_page.dart";
import "package:pyrin_app/ui.dart";
import "package:mobile_scanner/mobile_scanner.dart";

import "components/tokens_list.dart";

class BlobPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()
      ..color = Color.fromRGBO(199, 250, 112, 0.80)  // rgba(199, 250, 112, 0.80)
      ..style = PaintingStyle.fill;

    Path path = Path();
    path.moveTo(size.width * 1.015, size.height * 0.039);
    path.lineTo(size.width * 0.921, size.height * 0.044);
    path.cubicTo(
        size.width * 0.907, size.height * 0.032,
        size.width * 0.880, size.height * 0.006,
        size.width * 0.880, size.height * 0.008
    );
    path.cubicTo(
        size.width * 0.880, size.height * 0.010,
        size.width * 0.832, size.height * 0.061,
        size.width * 0.827, size.height * 0.064
    );
    path.cubicTo(
        size.width * 0.823, size.height * 0.066,
        size.width * 0.810, size.height * 0.083,
        size.width * 0.810, size.height * 0.086
    );
    path.cubicTo(
        size.width * 0.810, size.height * 0.089,
        size.width * 0.881, size.height * 0.137,
        size.width * 0.880, size.height * 0.140
    );
    path.cubicTo(
        size.width * 0.879, size.height * 0.143,
        size.width * 0.790, size.height * 0.194,
        size.width * 0.785, size.height * 0.197
    );
    path.cubicTo(
        size.width * 0.781, size.height * 0.200,
        size.width * 0.822, size.height * 0.218,
        size.width * 0.843, size.height * 0.227
    );
    path.cubicTo(
        size.width * 0.894, size.height * 0.246,
        size.width * 1.000, size.height * 0.284,
        size.width * 1.018, size.height * 0.287
    );
    path.cubicTo(
        size.width * 1.039, size.height * 0.292,
        size.width * 1.019, size.height * 0.223,
        size.width * 1.011, size.height * 0.207
    );
    path.cubicTo(
        size.width * 1.003, size.height * 0.194,
        size.width * 1.072, size.height * 0.197,
        size.width * 1.108, size.height * 0.201
    );
    path.lineTo(size.width * 1.073, size.height * 0.104);
    path.close();

    // Apply blur effect
    // final rect = path.getBounds();
    // canvas.saveLayer(rect, Paint());
    // canvas.drawPath(path, paint);
    //
    // final blur = 177.0;  // 177px blur
    // final sigma = blur * 0.57735 + 0.5;
    //
    // canvas.drawRect(
    //     rect,
    //     Paint()
    //     ..imageFilter = ui.ImageFilter.blur(sigmaX: sigma, sigmaY: sigma)
    //     ..blendMode = BlendMode.srcOver
    // );
    //
    canvas.drawShadow(path, Color.fromRGBO(199, 250, 112, 0.2), 125, true);

    canvas.restore();
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
class HomeSection extends StatefulWidget
{
    const HomeSection({super.key});

    @override
    State<HomeSection> createState() => HomeSectionState();
}

class HomeSectionState extends State<HomeSection>
{
    @override
    Widget build(BuildContext context)
    {
        final double width = MediaQuery.sizeOf(context).width;
        final double height = MediaQuery.sizeOf(context).height;
        double topHeight = height / 2.45;
        double bottomHeight = height - topHeight;

        return Scaffold(
            body: Center(
              child: SafeArea(
                child: Stack(
                  children: [
                    Container(
                      width: double.infinity,
                      child: SvgPicture.asset("assets/home_bg_circles.svg", fit: BoxFit.cover),
                    ),
                    // Positioned(
                    //   top: -125,
                    //   right: 0,
                    //   child: CustomPaint(
                    //     size: Size(390, 497),  // Original SVG size
                    //     painter: BlobPainter(),
                    //   ),
                    // ),
                    // Positioned(
                    //   right: 0,
                    //   child: Container(
                    //     width: 126,
                    //     height: 139,
                    //     child: CustomPaint(
                    //       painter: BlurredCirclePainter(
                    //         color: Color.fromRGBO(247, 112, 250, 0.35),
                    //         blurSigma: 177,
                    //       ),
                    //     ),
                    //   ),
                    // ),
                    Column(
                      children: [
                        Container(
                          padding: EdgeInsets.all(20),
                          height: topHeight,
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Column(
                                mainAxisAlignment: MainAxisAlignment.start,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: <Widget>[
                                  Container(
                                    child: Row(
                                      crossAxisAlignment: CrossAxisAlignment.center,
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      children: [
                                        IconButton(onPressed: onScanClick, icon: SvgPicture.asset("assets/icons/scan.svg")),
                                        Consumer<WalletProvider>(
                                          builder: (context, wallet, child)
                                          {
                                              Color statusColor = PyrinColors.ORANGE_COLOR;

                                              if (wallet.connectionState == WalletConnectionState.CONNECTED)
                                                statusColor = PyrinColors.GREEN_COLOR;
                                              else if (wallet.connectionState == WalletConnectionState.DISCONNECTED || wallet.connectionState == WalletConnectionState.ERROR)
                                                statusColor = PyrinColors.RED_COLOR;

                                              return Expanded(
                                                child: Row(
                                                  mainAxisAlignment: MainAxisAlignment.center,
                                                  children: [
                                                    Container(
                                                      width: 10,
                                                      height: 10,
                                                      decoration: BoxDecoration(
                                                        borderRadius: BorderRadius.circular(10),
                                                        color: statusColor,
                                                      ),
                                                    ),
                                                    Container(width: 6),
                                                    Text(
                                                        "Wallet (${wallet.receiveAddress.isNotEmpty ? AddressBook.shortenAddress(wallet.receiveAddress) : "N/A"})",
                                                        style: Theme.of(context).textTheme.bodySmall
                                                    ),
                                                  ],
                                                )
                                              );
                                          },
                                        ),
                                        IconButton(onPressed: (){}, icon: SvgPicture.asset("assets/icons/notification.svg")),
                                      ],
                                    ),
                                  ),
                                  SizedBox(height: 30),
                                  Container(
                                    width: 55,
                                    height: 55,
                                    alignment: Alignment.center,
                                    decoration: BoxDecoration(
                                      borderRadius: BorderRadius.circular(55),
                                      color: Colors.white.withOpacity(0.1),
                                    ),
                                    child: SvgPicture.asset("assets/pyrin-coin.svg", width: 32, height: 32),
                                  ),
                                  Consumer<WalletProvider>(
                                      builder: (context, wallet, child)
                                      {
                                          return Text(
                                              // NumberFormat("#,##0.00", "en_US").format(wallet.balance / 1e8),
                                              wallet.balance.toString(),
                                              style: Theme.of(context).textTheme.bodyLarge!.copyWith(fontSize: 36, fontWeight: FontWeight.w600)
                                          );
                                  }),
                                  PriceBadge(value: 5.01)
                                ],
                              ),
                              Column(
                                mainAxisAlignment: MainAxisAlignment.start,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: <Widget>[
                                  CircleIconButtonGroup(
                                    children: [
                                      CircleIconButton(icon: "send", text: "Send", onClick: onSendClick),
                                      CircleIconButton(icon: "receive", text: "Receive", onClick: onReceiveClick),
                                      CircleIconButton(icon: "swap", text: "Swap", onClick: onSwapClick),
                                    ],
                                  )
                                ],
                              ),
                            ],
                          ),
                        ),
                        Container(
                          padding: EdgeInsets.symmetric(horizontal: 20),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Text("Top Tokens", style: Theme.of(context).textTheme.bodySmall),
                              GestureDetector(
                                onTap: () => Navigator.pushNamed(context, "/tokens"),
                                child: Text("View all", style: Theme.of(context).textTheme.bodySmall),
                              ),
                            ],
                          ),
                        ),
                        Flexible(
                          child: Container(
                            padding: EdgeInsets.all(20),
                            child: TokensList(),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
        );
    }

    void onScanClick()
    {
      scanAddress(context, (address)
      {
          print("onScanClick: $address");
          Navigator.pushNamed(context, "/send", arguments: SendPageArguments(
            address: address,
          ));
      });
    }

    void onSendClick()
    {
        Navigator.pushNamed(context, "/send");
    }

    void onReceiveClick()
    {
      Navigator.pushNamed(context, "/receive");
    }

    void onSwapClick()
    {
      Provider.of<WalletProvider>(context, listen: false).connect();
    }
}

class CustomShapeWidget extends StatelessWidget {
  final double width;
  final double height;
  final Color color;
  final double radius;
  final double blurSigma;

  const CustomShapeWidget({
    Key? key,
    required this.width,
    required this.height,
    required this.color,
    this.radius = 35,
    this.blurSigma = 0,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      child: CustomPaint(
        painter: CustomShapePainter(
          color: color,
          radius: radius,
          blurSigma: blurSigma,
        ),
      ),
    );
  }
}

class CustomShapePainter extends CustomPainter {
  final Color color;
  final double radius;
  final double blurSigma;

  CustomShapePainter({required this.color, required this.radius, this.blurSigma = 0});

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill
      ..strokeWidth = 1.0;

    if (blurSigma > 0)
      paint.maskFilter = MaskFilter.blur(BlurStyle.normal, blurSigma);


    final r = 35.0;
    final y = r * 0.577142;

    final double width = size.width;
    final double height = size.height;
    // final double buttonWidth = 35 * 3;
    final double buttonWidth = r * 3;
    final double buttonHeight = height  - 11.2 * 2;

    final double curveWidth = 11.2;

    final double angle = 2 * math.asin(curveWidth / (2 * r));
    final double curveHeight = r * (1 - math.cos(angle / 2));

    final Path path = Path()
      ..moveTo(0 + radius, 0)

      //  ..lineTo((width - curveWidth) / 2, 0)
      // ..quadraticBezierTo(
      //   // width / 2, y * -3,
      //     width / 2, -curveHeight,
      //     (width + curveWidth) / 2, 0
      // )

      ..lineTo((width - curveWidth) / 2 - buttonWidth / 2, 0)
      ..quadraticBezierTo(
          width / 2, y * -3,
          (width + curveWidth) / 2 + buttonWidth / 2, 0
      )

      ..lineTo(width - radius, 0)
      ..quadraticBezierTo(width, 0, width, radius)
      ..lineTo(width, height - radius)
      ..quadraticBezierTo(width, height, width - radius, height)
      ..lineTo(radius, height)
      ..quadraticBezierTo(0, height, 0, height - radius)
      ..lineTo(0, radius)
      ..quadraticBezierTo(0, 0, radius, 0);

    canvas.drawPath(path, paint);

    canvas.drawCircle(ui.Offset(width / 2, y), r, paint); // 20.2 / 35
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}

class SVGPathClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    // This is where we'll define our SVG path
    Path path = parseSVGPath(
        "M113.941 32.7413C125.332 32.7154 134.87 24.6679 142.381 16.1028C150.792 6.50981 163.178 0.433804 177.001 0.402376C190.825 0.370947 203.238 6.39056 211.693 15.9453C219.242 24.4761 228.817 32.4801 240.208 32.4542L328.072 32.2545C342.432 32.2218 354.099 43.8359 354.131 58.1953L354.198 87.5407C354.231 101.9 342.617 113.567 328.257 113.6L26.2581 114.286C11.8987 114.319 0.231682 102.705 0.199035 88.3456L0.132314 59.0002C0.0996664 44.6408 11.7138 32.9737 26.0731 32.9411L113.941 32.7413Z"
    );

    // Scale the path to fit the size
    final scaleX = size.width / 355;
    final scaleY = size.height / 115;
    final scaledPath = path.transform(Matrix4.diagonal3Values(scaleX, scaleY, 1).storage);

    return scaledPath;
  }

  @override
  bool shouldReclip(CustomClipper<Path> oldClipper) => false;

  Path parseSVGPath(String svgPath) {
    final path = Path();
    final coordinates = svgPath.replaceAll(RegExp(r'[A-Za-z]'), ' \$0 ').trim().split(RegExp(r'\s+'));
    var currentX = 0.0;
    var currentY = 0.0;

    for (var i = 0; i < coordinates.length; i++) {
      switch (coordinates[i]) {
        case 'M':
          currentX = double.parse(coordinates[++i]);
          currentY = double.parse(coordinates[++i]);
          path.moveTo(currentX, currentY);
          break;
        case 'C':
          final x1 = double.parse(coordinates[++i]);
          final y1 = double.parse(coordinates[++i]);
          final x2 = double.parse(coordinates[++i]);
          final y2 = double.parse(coordinates[++i]);
          final x = double.parse(coordinates[++i]);
          final y = double.parse(coordinates[++i]);
          path.cubicTo(x1, y1, x2, y2, x, y);
          currentX = x;
          currentY = y;
          break;
        case 'H':
          final x = double.parse(coordinates[++i]);
          path.lineTo(x, currentY);
          currentX = x;
          break;
        case 'V':
          final y = double.parse(coordinates[++i]);
          path.lineTo(currentX, y);
          currentY = y;
          break;
        case 'Z':
          path.close();
          break;
      }
    }

    return path;
  }
}
