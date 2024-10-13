import "dart:ui" as ui;
import "package:flutter/material.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:pyrin_app/ui.dart";

enum Section
{
    HOME,
    SWAP,
    ASSETS,
    MENU,
}

class PyrinNavigationBar extends StatelessWidget
{
    Section section;
    Function(Section section) onSectionChanged;

    PyrinNavigationBar({required this.section, required this.onSectionChanged});

    @override
    Widget build(BuildContext context)
    {
      double width = MediaQuery.sizeOf(context).width;
      // double height = MediaQuery.sizeOf(context).height;
      // final double height = 115;
      final double height = 115 / 411 * width;
      final double buttonSize = height * 0.615;
      // final double buttonSize = width * 0.18;

      return Container(
        width: double.infinity,
        height: height,
        margin: EdgeInsets.all(20),

        child: Stack(
          alignment: Alignment.bottomCenter,
          children: [
            CustomShapeWidget2(
              width: width,
              height: height,
              // color: Colors.black,
              color: Colors.white.withOpacity(0.01),
              // color: Colors.transparent,
            ),
            ClipPath(
              clipper: CustomNavigationShapeClipper(),
              child: BackdropFilter(
                filter: ui.ImageFilter.blur(sigmaX: 17.5, sigmaY: 17.5),
                child: Container(
                  width: width,
                  height: height,
                  decoration: BoxDecoration(
                    // color: Colors.white.withOpacity(0.1),
                    // color: Color(0xff161816),
                    color: Colors.white.withOpacity(0.01),
                    borderRadius: BorderRadius.circular(35),
                  ),
                ),
              ),
            ),
            Container(
              width: width,
              height: height * 0.725,
              padding: EdgeInsets.all(20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(30),
                // color: Colors.blue.withOpacity(0.5),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  _buildIconButton(context: context, section: Section.HOME, icon: "home", name: "Home"),
                  _buildIconButton(context: context, section: Section.SWAP, icon: "primary-swap", name: "Swap"),
                  Expanded(child: Container()),
                  _buildIconButton(context: context, section: Section.ASSETS, icon: "assets", name: "Assets"),
                  _buildIconButton(context: context, section: Section.MENU, icon: "menu", name: "Menu"),
                ],
              ),
            ),
            Align(
              alignment: Alignment.topCenter,
              child: Container(
                margin: EdgeInsets.only(top: 12),
                child: PyrinPrimaryCircleButton(
                  size: buttonSize,
                  icon: "add",
                  onClick: ()
                  {
                      Navigator.pushNamed(context, "/token/create");
                  },
                ),
              ),
            )
          ],
        ),

        // child: CustomShapeWidget2(
        //   width: width,
        //   height: 115,
        //   color: Colors.blue,
        // ),

        // child: CustomShape(width: width, height: 150),
        // child: CustomShapeWidget(
        //   width: 354,
        //   height: 82,
        //   color: Colors.blue,
        // ),

        // child: Container(
        //   width: double.infinity,
        //   height: 80,
        //   child: Stack(
        //     children: [
        //       CustomShapeWidget(
        //         width: double.infinity,
        //         height: 80,
        //         color: Colors.white.withOpacity(0.1),
        //       ),
        //       CustomShapeWidget(
        //         width: double.infinity,
        //         height: 80,
        //         blurSigma: 17.5,
        //         color: Colors.white.withOpacity(0.01),
        //       )
        //     ],
        //   ),
        // ),

        // child: Container(
        //   width: 355,
        //   height: 115,
        //   color: Colors.red,
        //   child: ClipPath(
        //     clipper: ShapeBorderClipper(
        //       shape: RoundedRectangleBorder(
        //         borderRadius: BorderRadius.circular(35),
        //       ),
        //     ),
        //     child: BackdropFilter(
        //       filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        //       child: Container(
        //         width: 354,
        //         height: 81.3462,
        //         color: Colors.transparent,
        //       ),
        //     ),
        //   ),
        // ),
        // child: Container(
        //   width: double.infinity,
        //   height: 82,
        //   child: SvgPicture.asset("assets/home_nav_bg.svg", fit: BoxFit.contain),
        // ),
        // child: Container(
        //   width: 50,
        //   height: 50,
        //   child: BackdropFilter(
        //     filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        //     child: Container(
        //       width: double.infinity,
        //       height: 82,
        //       decoration: BoxDecoration(
        //         color: Colors.white.withOpacity(0.1),
        //         borderRadius: BorderRadius.circular(35),
        //       ),
        //       child: SvgPicture.asset("assets/navigation_bar.svg", fit: BoxFit.cover),
        //     ),
        //   ),
        // ),
      );
    }

    Widget _buildIconButton({
      required BuildContext context,
      required Section section,
      required String icon,
      required String name
    })
    {
      final active = this.section == section;
      double width = MediaQuery.sizeOf(context).width;
      // double height = MediaQuery.sizeOf(context).height;
      // final double height = 115;
      final double size = 115 / 411 * width * 0.5;
      final Color color = active ? PyrinColors.TEXT_COLOR : PyrinColors.TEXT_COLOR.withOpacity(0.4);

      return InkWell(
        onTap: () => onSectionChanged(section),
        child: Container(
          width: size,
          height: size,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SvgPicture.asset(
                "assets/icons/$icon.svg",
                width: size * 0.4,
                height: size * 0.4,
                colorFilter: ColorFilter.mode(
                    color,
                    BlendMode.srcIn
                ),
              ),
              SizedBox(height: 5),
              Text(name, style: Theme.of(context).textTheme.bodySmall!.copyWith(color: color, fontSize: 11)),
            ],
          ),
          // child: IconButton(
          //   onPressed: (){},
          //   icon: Column(
          //     mainAxisSize: MainAxisSize.min,
          //     children: [
          //       SvgPicture.asset(
          //         "assets/icons/$icon.svg",
          //         colorFilter: ColorFilter.mode(
          //             color,
          //             BlendMode.srcIn
          //         ),
          //       ),
          //       SizedBox(height: 5),
          //       Text("asdasd", style: Theme.of(context).textTheme.bodySmall!.copyWith(color: color)),
          //     ],
          //   ),
          // ),
        ),
      );
    }
}



class CustomShapeWidget2 extends StatelessWidget {
  final double width;
  final double height;
  final Color color;

  const CustomShapeWidget2({
    Key? key,
    this.width = 355,
    this.height = 115,
    required this.color,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size(width, height),
      painter: CustomShapePainter2(color: color),
    );
  }
}

Path getNavigationMenuPath(Size size)
{
  final Path path = Path();

  // Convert absolute coordinates to relative
  path.moveTo(size.width * (113.941 / 355), size.height * (32.7413 / 115));

  double offset = 1;

  path.cubicTo(
      size.width * (125.332 / 355), size.height * 1 * (32.7154 / 115),
      size.width * (134.87 / 355), size.height * offset * (24.6679 / 115),
      size.width * (142.381 / 355), size.height * offset * (16.1028 / 115)
  );
  path.cubicTo(
      size.width * (150.792 / 355), size.height * offset * (6.50981 / 115),
      size.width * (163.178 / 355), size.height * offset * (0.433804 / 115),
      size.width * (177.001 / 355), size.height * offset * (0.402376 / 115)
  );
  path.cubicTo(
      size.width * (190.825 / 355), size.height * offset * (0.370947 / 115),
      size.width * (203.238 / 355), size.height * offset * (6.39056 / 115),
      size.width * (211.693 / 355), size.height * offset * (15.9453 / 115)
  );
  path.cubicTo(
      size.width * (219.242 / 355), size.height * offset * (24.4761 / 115),
      size.width * (228.817 / 355), size.height * 1 * (32.4801 / 115),
      size.width * (240.208 / 355), size.height * 1 * (32.4542 / 115)
  );

  path.lineTo(size.width * (328.072 / 355), size.height * (32.2545 / 115));
  path.cubicTo(
      size.width * (342.432 / 355), size.height * (32.2218 / 115),
      size.width * (354.099 / 355), size.height * (43.8359 / 115),
      size.width * (354.131 / 355), size.height * (58.1953 / 115)
  );
  path.lineTo(size.width * (354.198 / 355), size.height * (87.5407 / 115));
  path.cubicTo(
      size.width * (354.231 / 355), size.height * (101.9 / 115),
      size.width * (342.617 / 355), size.height * (113.567 / 115),
      size.width * (328.257 / 355), size.height * (113.6 / 115)
  );
  path.lineTo(size.width * (26.2581 / 355), size.height * (114.286 / 115));
  path.cubicTo(
      size.width * (11.8987 / 355), size.height * (114.319 / 115),
      size.width * (0.231682 / 355), size.height * (102.705 / 115),
      size.width * (0.199035 / 355), size.height * (88.3456 / 115)
  );
  path.lineTo(size.width * (0.132314 / 355), size.height * (59.0002 / 115));
  path.cubicTo(
      size.width * (0.0996664 / 355), size.height * (44.6408 / 115),
      size.width * (11.7138 / 355), size.height * (32.9737 / 115),
      size.width * (26.0731 / 355), size.height * (32.9411 / 115)
  );
  path.close();

  return path;
}

class CustomNavigationShapeClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    return getNavigationMenuPath(size);
  }

  @override
  bool shouldReclip(covariant CustomClipper<Path> oldClipper) {
    return false;
  }

}

class CustomShapePainter2 extends CustomPainter {
  final Color color;

  CustomShapePainter2({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final Path path = getNavigationMenuPath(size);

    // canvas.drawColor(Color(0), BlendMode.clear);

    canvas.drawPath(path, paint);

    canvas.drawShadow(path.shift(Offset(0.5, -2)), Color.fromRGBO(255, 255, 255, 0.15), 1, false);
    canvas.drawShadow(path.shift(Offset(-0.5, -2)), Color.fromRGBO(255, 255, 255, 0.15), 1, false);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

